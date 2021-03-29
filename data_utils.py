from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torchvision import transforms, datasets

from rich_utils import RICHDataProvider

n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()


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


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=False,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_RICH(particle, drop_weights, dataroot, download):
    flow_shape = (5,)

    # TODO: rewrite rich utils to make it use given path
    path = Path(dataroot) / "data" / "data_calibsample"
    # TODO: download dataset if needed

    train_data, test_data, scaler = RICHDataProvider().get_merged_typed_dataset(
        particle, dtype=np.float32, log=True
    )

    condition_columns = ["Brunel_P", "Brunel_ETA", "nTracks_Brunel"]
    flow_columns = ["RichDLLe", "RichDLLk", "RichDLLmu", "RichDLLp", "RichDLLbt"]
    weight_column = "probe_sWeight"

    if drop_weights:
        train_dataset = TensorDataset(
            torch.from_numpy(train_data[flow_columns].values),
            torch.from_numpy(train_data[condition_columns].values),
            torch.from_numpy(train_data[weight_column].values),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_data[flow_columns].values),
            torch.from_numpy(test_data[condition_columns].values),
            torch.from_numpy(test_data[weight_column].values),
        )
    else:
        train_dataset = TensorDataset(
            torch.from_numpy(train_data[flow_columns].values),
            torch.from_numpy(train_data[condition_columns].values),
            torch.from_numpy(train_data[weight_column].values),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(test_data[flow_columns].values),
            torch.from_numpy(test_data[condition_columns].values),
            torch.from_numpy(test_data[weight_column].values),
        )

    return flow_shape, len(condition_columns), train_dataset, test_dataset, scaler
