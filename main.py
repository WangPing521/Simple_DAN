import sys

sys.path.insert(0, "../")

from deepclustering2.configparser import ConfigManger
from deepclustering2.dataset import ACDCSemiInterface
from deepclustering2.dataset.segmentation import ProstateSemiInterface, SpleenSemiInterface
from deepclustering2.utils import set_benchmark
from deepclustering.schedulers import Weight_RampScheduler
from lossfunc.augment import val_transform, train_transform
from lossfunc.augment_spleen import val_transformS, train_transformS
from trainers.DANTrainer import DANTrainer
from networks.disc import OfficialDiscriminator
from networks.enet import Enet

config = ConfigManger("config/config.yaml").config
set_benchmark(config['seed'])

model = Enet(input_dim=1, num_classes=4)
discriminator = OfficialDiscriminator(nc=5, ndf=64)

if config['Dataset'] == 'acdc':
    dataset_handler = ACDCSemiInterface(**config["Data"])
elif config['Dataset'] == 'spleen':
    dataset_handler = SpleenSemiInterface(**config["Data"])
    train_transform = train_transformS
    val_transform = val_transformS
elif config['Dataset'] == 'prostate':
    dataset_handler = ProstateSemiInterface(**config["Data"])


def get_group_set(dataloader):
    return set(sorted(dataloader.dataset.get_group_list()))


dataset_handler.compile_dataloader_params(**config["DataLoader"])
label_loader, unlab_loader, val_loader = dataset_handler.SemiSupervisedDataLoaders(
    labeled_transform=train_transform,
    unlabeled_transform=train_transform,
    val_transform=val_transform,
    group_val=True,
    use_infinite_sampler=True,
)
assert get_group_set(label_loader) & get_group_set(unlab_loader) == set()
assert (get_group_set(label_loader) | get_group_set(unlab_loader)) & get_group_set(val_loader) == set()
print(
    f"Labeled loader with {len(get_group_set(label_loader))} groups: \n {', '.join(sorted(get_group_set(label_loader))[:5])}"
)
print(
    f"Unabeled loader with {len(get_group_set(unlab_loader))} groups: \n {', '.join(sorted(get_group_set(unlab_loader))[:5])}"
)
print(
    f"Val loader with {len(get_group_set(val_loader))} groups: \n {', '.join(sorted(get_group_set(val_loader))[:5])}"
)

RegScheduler = Weight_RampScheduler(**config["RegScheduler"])

trainer = DANTrainer(
    model=model,
    discriminator=discriminator,
    lab_loader=label_loader,
    unlab_loader=unlab_loader,
    weight_scheduler=RegScheduler,
    val_loader=val_loader,
    config=config,
    **config["Trainer"]
)
trainer.start_training()
