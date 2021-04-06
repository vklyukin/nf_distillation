import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms, datasets
from data.src.utils import preprocess


def get_CelebA(augment, dataroot, download):
    image_shape = (64, 64, 3)
    num_classes = 40

    test_transform = transforms.Compose(
        [transforms.Resize(image_shape[:-1]), transforms.ToTensor(), preprocess]
    )

    if augment:
        transformations = [
            transforms.Resize(image_shape[:-1]),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = [
            transforms.Resize(image_shape[:-1]),
        ]

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    path = Path(dataroot) / "data" / "celeba"
    train_dataset = datasets.CelebA(
        path, split="train", transform=train_transform, download=download,
    )

    test_dataset = datasets.CelebA(
        path, split="valid", transform=test_transform, download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset
