import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset


def get_BSDS300(dataroot):
    path = Path(dataroot) / "data" / "BSDS300" / "BSDS300.hdf5"

    data = h5py.File(path)
    train, val = data["train"].astype(np.float32), data["validation"].astype(np.float32)

    train_dataset = TensorDataset(torch.from_numpy(train))
    val_dataset = TensorDataset(
        torch.from_numpy(val),
    )

    flow_shape = (train.shape[1],)

    return flow_shape, 0, train_dataset, val_dataset
