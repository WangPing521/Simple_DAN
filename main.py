import sys

sys.path.insert(0, "../")

from deepclustering2.configparser import ConfigManger
from deepclustering2.dataset import ACDCSemiInterface
from deepclustering2.dataset.segmentation import ProstateSemiInterface, SpleenSemiInterface
from deepclustering2.utils import fix_all_seed
from deepclustering.schedulers import Weight_RampScheduler
from lossfunc.augment import val_transform, train_transform
from lossfunc.augment_spleen import val_transformS, train_transformS
from lossfunc.helper import ModelList
from lossfunc.models import Model
from trainers.DANTrainer import DANTrainer

config = ConfigManger("config/config.yaml").config
fix_all_seed(config['seed'])


model1 = Model(config["Arch1"], config["Optim"], config["Scheduler"])
model2 = Model(config["Arch2"], config["Optim"], config["Scheduler"])

models = ModelList([model1, model2])

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
    model=models,
    lab_loader=label_loader,
    unlab_loader=unlab_loader,
    weight_scheduler=RegScheduler,
    val_loader=val_loader,
    config=config,
    **config["Trainer"]
)
trainer.start_training()
