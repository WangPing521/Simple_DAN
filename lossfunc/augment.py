import PIL

from deepclustering.augment import pil_augment, SequentialWrapper
from torchvision import transforms

image_transform = pil_augment.Compose(
    [
        pil_augment.RandomRotation(degrees=45, resample=PIL.Image.BILINEAR),
        pil_augment.RandomCrop((224, 224), fill=(0,)),
        pil_augment.RandomChoice(
            [pil_augment.RandomVerticalFlip(), pil_augment.RandomHorizontalFlip()]
        ),
        transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
        pil_augment.ToTensor(),
    ]
)
label_transform = pil_augment.Compose(
    [
        pil_augment.RandomRotation(degrees=45, resample=PIL.Image.NEAREST),
        pil_augment.RandomCrop((224, 224), fill=(0,)),
        pil_augment.RandomChoice(
            [pil_augment.RandomVerticalFlip(), pil_augment.RandomHorizontalFlip()]
        ),
        pil_augment.ToLabel(),
    ]
)
val_img_transform = pil_augment.Compose(
    [pil_augment.CenterCrop((224, 224), ), pil_augment.ToTensor()]
)
val_target_transform = pil_augment.Compose(
    [pil_augment.CenterCrop((224, 224), ), pil_augment.ToLabel()]
)
train_transform = SequentialWrapper(
    image_transform, label_transform, if_is_target=[False, True]
)
val_transform = SequentialWrapper(
    val_img_transform, val_target_transform, if_is_target=[False, True]
)
