import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import TensorDataset


def load_data(root_path):

    data = np.load(root_path)
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):

    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


def get_MINIBOONE(dataroot):
    path = Path(dataroot) / "data" / "miniboone" / "data.npy"

    train, val, _ = load_data_normalised(path)

    train_dataset = TensorDataset(torch.from_numpy(train.astype(np.float32)))
    val_dataset = TensorDataset(torch.from_numpy(val.astype(np.float32)))

    flow_shape = (train.shape[1],)

    return flow_shape, 0, train_dataset, val_dataset
