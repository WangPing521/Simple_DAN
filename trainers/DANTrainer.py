import os
from typing import Union

import numpy as np
import torch
from deepclustering.trainer import _Trainer
from deepclustering2.dataloader.dataloader import _BaseDataLoaderIter
from deepclustering2.loss import KL_div, Entropy
from deepclustering2.schedulers import Weight_RampScheduler
from deepclustering2.utils import tqdm_, flatten_dict, nice_dict, class2one_hot, Path
from lossfunc.helper import average_list, merge_input
from meters import UniversalDice
from meters.averagemeter import AverageValueMeter
from torch import nn, optim
from torch.utils.data import DataLoader


class DANTrainer(_Trainer):
    this_directory = os.path.abspath(os.path.dirname(__file__))
    PROJECT_PATH = os.path.dirname(this_directory)
    RUN_PATH = str(Path(PROJECT_PATH, "runs"))

    unlabeled_tag = 0
    labeled_tag = 1

    def __init__(
        self,
        model: nn.Module,
        discriminator: nn.Module,
        lab_loader: Union[DataLoader, _BaseDataLoaderIter],
        unlab_loader: Union[DataLoader, _BaseDataLoaderIter],
        val_loader: DataLoader,
        weight_scheduler: Weight_RampScheduler = None,
        max_epoch: int = 100,
        save_dir: str = "base",
        checkpoint_path: str = None,
        device="cpu",
        config: dict = None,
        num_batches=100,
        *args,
        **kwargs,
    ) -> None:
        self._lab_loader = lab_loader
        self._unlab_loader = unlab_loader
        super().__init__(
            model,
            None,
            val_loader,
            max_epoch,
            save_dir,
            checkpoint_path,
            device,
            config,
            *args,
            **kwargs,
        )
        self._num_batches = num_batches
        self._ce_criterion = KL_div(verbose=False)
        self._weight_scheduler = weight_scheduler
        self.checkpoint_path = checkpoint_path
        self._entropy_criterion = Entropy()
        self._disc = discriminator

        self._model_optimizer = optim.Adam(model.parameters(), lr=1e-5)
        self._disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-5)
        self._bce_criterion = nn.BCELoss()

    def register_meters(self, enable_drawer=True) -> None:
        super(DANTrainer, self).register_meters()
        c = self._config['Arch1'].get('num_classes')
        report_axises = list(range(1, c))

        self._meter_interface.register_new_meter(
            f"tra_dice", UniversalDice(C=c, report_axises=report_axises), group_name="train"
        )
        self._meter_interface.register_new_meter(
            f"val_dice", UniversalDice(C=c, report_axises=report_axises), group_name="val"
        )

        self._meter_interface.register_new_meter(
            "sup_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "D(L)", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "D(U)1", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "D(U)2", AverageValueMeter(), group_name="train"
        )

        self._meter_interface.register_new_meter(
            "adv_weight", AverageValueMeter(), group_name="train"
        )

    def _train_loop(self,
                    lab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                    unlab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                    epoch: int = 0,
                    *args,
                    **kwargs):
        self._model.train()
        batch_indicator = tqdm_(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")
        for batch_id, lab_data, unlab_data in zip(batch_indicator, lab_loader, unlab_loader):
            self._run_step(lab_data=lab_data, unlab_data=unlab_data)
            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("train")
                report_statue = {k: v for k, v in flatten_dict(report_statue).items() if not np.isnan(v)}
                batch_indicator.set_postfix(report_statue)
        report_statue = self._meter_interface.tracking_status("train")
        report_statue = {k: v for k, v in flatten_dict(report_statue).items() if not np.isnan(v)}
        batch_indicator.set_postfix(report_statue)
        self.writer.add_scalar_with_tag(
            "train", report_statue, global_step=epoch
        )
        print(f"Training Epoch {epoch}: {nice_dict(flatten_dict(report_statue))}")

    def _eval_loop(self,
                   val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                   epoch: int = 0,
                   *args,
                   **kwargs,
                   ):
        self._model.eval()
        val_indicator = tqdm_(val_loader)
        val_indicator.set_description(f"Validation Epoch {epoch:03d}")
        for batch_id, data in enumerate(val_indicator):
            image, target, filename = (
                data[0][0].to(self._device),
                data[0][1].to(self._device),
                data[1],
            )
            preds = self._model(image).softmax(1)
            self._meter_interface['val_dice'].add(
                preds.max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:-2]) for x in filename]
            )
            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("val")
                val_indicator.set_postfix(flatten_dict(report_statue))
        report_statue = self._meter_interface.tracking_status("val")
        val_indicator.set_postfix(flatten_dict(report_statue))
        self.writer.add_scalar_with_tag(
            "val", flatten_dict(report_statue), global_step=epoch
        )
        print(f"Validation Epoch {epoch}: {nice_dict(flatten_dict(report_statue))}")

        return average_list(self._meter_interface[f"val_dice"].summary().values())

    def schedulerStep(self):
        self._weight_scheduler.step()

    def _start_training(self):
        from deepclustering2.optim import get_lrs_from_optimizer
        self.to(self._device)
        for epoch in range(self._start_epoch, self._max_epoch):
            self._meter_interface['lr'].add(get_lrs_from_optimizer(self._model_optimizer)[0])
            self._meter_interface['adv_weight'].add(self._weight_scheduler.value)

            self.train_loop(
                lab_loader=self._lab_loader,
                unlab_loader=self._unlab_loader,
                epoch=epoch
            )
            with torch.no_grad():
                current_score = self.eval_loop(self._val_loader, epoch)
            self.schedulerStep()
            self.save_checkpoint(self.state_dict(), epoch, current_score)
            self._meter_interface.summary().to_csv(self._save_dir / "wholeMeter.csv")

    def _run_step(self, lab_data, unlab_data, *args, **kwargs):
        adv_weight = self._weight_scheduler.value
        self._meter_interface["adv_weight"].add(adv_weight)

        image, target, filename = (
            lab_data[0][0].to(self._device),
            lab_data[0][1].to(self._device),
            lab_data[1],
        )
        onehot_target = class2one_hot(
            target.squeeze(1), self._model.num_classes
        )
        uimage = unlab_data[0][0].to(self._device)

        # train segmentor
        lab_preds = self._model(image).softmax(1)
        unlab_preds = self._model(uimage).softmax(1)

        # offer information
        b_label = image.shape[0]
        b_unlabeled = uimage.shape[0]

        # supervised learning
        self._model_optimizer.zero_grad()

        sup_loss = self._ce_criterion(lab_preds, onehot_target)
        if torch.isnan(sup_loss):
            raise RuntimeError("nan")
        self._meter_interface["sup_loss"].add(sup_loss.item())
        self._meter_interface['tra_dice'].add(
            lab_preds.max(1)[1],
            target.squeeze(1),
            group_name=["_".join(x.split("_")[:-2]) for x in filename]
        )
        gen_loss = torch.tensor(0.0, dtype=torch.float, device=self._device)
        if adv_weight > 0:
            # train generator
            unlab_decision = self._disc(merge_input(pred=unlab_preds, img=uimage)).squeeze()
            self._meter_interface["D(U)2"].add(unlab_decision.mean().item())
            labeled_targt_ = torch.zeros(b_unlabeled, device=self._device).fill_(self.labeled_tag)
            gen_loss = self._bce_criterion(unlab_decision, labeled_targt_)
            if torch.isnan(gen_loss):
                raise RuntimeError("nan")

        segmentor_loss = sup_loss + adv_weight * gen_loss
        segmentor_loss.backward()
        self._model_optimizer.step()

        # adversarial learning
        # train discriminator
        if adv_weight > 0:
            self._disc_optimizer.zero_grad()
            with torch.no_grad():
                lab_preds = self._model(image).softmax(1)
                unlab_preds = self._model(uimage).softmax(1)

            labeled_targt_ = torch.zeros(b_label, device=self._device).fill_(self.labeled_tag)
            unlabeled_target_ = torch.zeros(b_unlabeled, device=self._device).fill_(self.unlabeled_tag)

            lab_decision = self._disc(merge_input(pred=lab_preds.detach(), img=image)).squeeze()
            unlab_decision = self._disc(merge_input(pred=unlab_preds.detach(), img=uimage)).squeeze()
            self._meter_interface["D(L)"].add(lab_decision.mean().item())
            self._meter_interface["D(U)1"].add(unlab_decision.mean().item())

            labeled_loss = self._bce_criterion(lab_decision, labeled_targt_)
            unlabeled_loss = self._bce_criterion(unlab_decision, unlabeled_target_)
            if torch.isnan(labeled_loss) or torch.isnan(unlabeled_loss):
                raise RuntimeError("loss nan")

            disc_loss = labeled_loss + unlabeled_loss
            (disc_loss * adv_weight).backward()
            self._disc_optimizer.step()

    def to(self, device):
        self._model.to(device)
        self._disc.to(device)
