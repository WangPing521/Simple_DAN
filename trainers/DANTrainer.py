from typing import Union
import os
import torch
from deepclustering2 import ModelMode
from deepclustering2.dataloader.dataloader import _BaseDataLoaderIter
from deepclustering2.loss import SimplexCrossEntropyLoss, Entropy
from deepclustering2.models import ZeroGradientBackwardStep
from deepclustering.trainer import _Trainer
from deepclustering.schedulers import Weight_RampScheduler
from deepclustering2.utils import tqdm_, flatten_dict, nice_dict, class2one_hot, Path
from torch.utils.data import DataLoader

from lossfunc.helper import ModelList, average_list, merge_input
from meters import UniversalDice
from meters.averagemeter import AverageValueMeter


class DANTrainer(_Trainer):
    this_directory = os.path.abspath(os.path.dirname(__file__))
    PROJECT_PATH = os.path.dirname(this_directory)
    RUN_PATH = str(Path(PROJECT_PATH, "runs"))

    def __init__(
            self,
            model: ModelList,
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
        self._ce_criterion = SimplexCrossEntropyLoss()
        self._weight_scheduler = weight_scheduler
        self.checkpoint_path = checkpoint_path
        self._entropy_criterion = Entropy()

    def register_meters(self, enable_drawer=True) -> None:
        super(DANTrainer, self).register_meters()
        c = self._config['Arch1'].get('num_classes')
        report_axises = []
        for axi in range(c):
            report_axises.append(axi)

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
            "SN_loss", AverageValueMeter(), group_name="train"
        )
        self._meter_interface.register_new_meter(
            "EN_loss", AverageValueMeter(), group_name="train"
        )

        self._meter_interface.register_new_meter(
            "adv_weight", AverageValueMeter(), group_name="train"
        )

    def _train_loop(self,
                    lab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                    unlab_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                    epoch: int = 0,
                    mode=ModelMode.TRAIN,
                    *args,
                    **kwargs):
        self._model.set_mode(mode)
        batch_indicator = tqdm_(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")
        for batch_id, lab_data, unlab_data in zip(batch_indicator, lab_loader, unlab_loader):
            loss, lab_loss, ENunlab_loss, SNunlab_loss = self._run_step(lab_data=lab_data, unlab_data=unlab_data)
            with ZeroGradientBackwardStep(
                    self._weight_scheduler.value * (lab_loss + ENunlab_loss),
                    self._model[1]
            ) as EN_loss:
                EN_loss.backward(retain_graph=True)
            with ZeroGradientBackwardStep(
                    loss + self._weight_scheduler.value * SNunlab_loss,
                    self._model[0]
            ) as SN_loss:
                SN_loss.backward(retain_graph=True)
            self._meter_interface['EN_loss'].add(EN_loss.item())
            self._meter_interface['sup_loss'].add(loss.item())
            self._meter_interface['SN_loss'].add(SN_loss.item())
            if ((batch_id + 1) % 5) == 0:
                report_statue = self._meter_interface.tracking_status("train")
                batch_indicator.set_postfix(flatten_dict(report_statue))
        report_statue = self._meter_interface.tracking_status("train")
        batch_indicator.set_postfix(flatten_dict(report_statue))
        self.writer.add_scalar_with_tag(
            "train", flatten_dict(report_statue), global_step=epoch
        )
        print(f"Training Epoch {epoch}: {nice_dict(flatten_dict(report_statue))}")

    def _eval_loop(self,
                   val_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
                   epoch: int = 0,
                   mode=ModelMode.EVAL,
                   *args,
                   **kwargs,
                   ):
        self._model.set_mode(mode)
        val_indicator = tqdm_(val_loader)
        val_indicator.set_description(f"Validation Epoch {epoch:03d}")
        for batch_id, data in enumerate(val_indicator):
            image, target, filename = (
                data[0][0].to(self._device),
                data[0][1].to(self._device),
                data[1],
            )
            preds = self._model[0](image).softmax(1)
            self._meter_interface['val_dice'].add(
                preds.max(1)[1],
                target.squeeze(1),
                group_name=["_".join(x.split("_")[:-2]) for x in filename]
            )
            # save_images(preds.max(1)[1], names=filename, root=self._config['Trainer']['save_dir'], mode='prediction', iter=epoch)
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

        for segmentator in self._model:
            segmentator.schedulerStep()
        self._weight_scheduler.step()

    def _start_training(self):
        for epoch in range(self._start_epoch, self._max_epoch):
            if self._model.get_lr() is not None:
                self._meter_interface['lr'].add(self._model.get_lr()[0])
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
        image, target, filename = (
            lab_data[0][0].to(self._device),
            lab_data[0][1].to(self._device),
            lab_data[1],
        )
        onehot_target = class2one_hot(
            target.squeeze(1), self._model[0]._torchnet.num_classes
        )
        uimage = unlab_data[0][0].to(self._device)
        lab_preds = self._model[0](image).softmax(1)
        unlab_preds = self._model[0](uimage).softmax(1)

        self._meter_interface['tra_dice'].add(
            lab_preds.max(1)[1],
            target.squeeze(1),
            group_name=["_".join(x.split("_")[:-2]) for x in filename]
        )

        loss = self._ce_criterion(lab_preds, onehot_target)

        lab_decision = self._model[1](merge_input(pred=lab_preds, img=image)).softmax(1)
        unlab_decision = self._model[1](merge_input(pred=unlab_preds, img=uimage)).softmax(1)

        lab_loss = (lab_decision[:, 0] + 1e-16).log()
        lab_loss = (-1.0 * lab_loss.sum()) / (lab_decision[:, 0].shape[0])

        ENunlab_loss = (unlab_decision[:, 1] + 1e-16).log()
        ENunlab_loss = (-1.0 * ENunlab_loss.sum()) / (unlab_decision[:, 1].shape[0])

        SNunlab_loss = (unlab_decision[:, 0] + 1e-16).log()
        SNunlab_loss = (-1.0 * SNunlab_loss.sum()) / (unlab_decision[:, 0].shape[0])
        return loss, lab_loss, ENunlab_loss, SNunlab_loss
