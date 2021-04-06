import numpy as np
import os
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from time import time
from torch.utils.data import TensorDataset


data_dir = "/home/vdklyukin/nf_distillation/data/data/data_calibsample/"


class RICHDataProvider:
    def __init__(self):
        self.list_particles = ["kaon", "pion", "proton", "muon", "electron"]

        self.datasets = {
            particle: self.get_particle_dset(particle)
            for particle in self.list_particles
        }

        self.dll_columns = [
            "RichDLLe",
            "RichDLLk",
            "RichDLLmu",
            "RichDLLp",
            "RichDLLbt",
        ]
        self.raw_feature_columns = ["Brunel_P", "Brunel_ETA", "nTracks_Brunel"]
        self.weight_col = "probe_sWeight"

        self.y_count = len(self.dll_columns)
        self.TEST_SIZE = 0.5

    def get_particle_dset(self, particle):
        return [data_dir + name for name in os.listdir(data_dir) if particle in name]

    def load_and_cut(self, file_name):
        data = pd.read_csv(file_name, delimiter="\t")
        return data[self.dll_columns + self.raw_feature_columns + [self.weight_col]]

    def load_and_merge_and_cut(self, filename_list):
        return pd.concat(
            [
                self.load_and_cut(os.path.join(os.getcwd(), fname))
                for fname in filename_list
            ],
            axis=0,
            ignore_index=True,
        )

    def split(self, data):
        data_train, data_val = train_test_split(
            data, test_size=self.TEST_SIZE, random_state=42
        )
        data_val, data_test = train_test_split(
            data_val, test_size=self.TEST_SIZE, random_state=1812
        )
        return (
            data_train.reset_index(drop=True),
            data_val.reset_index(drop=True),
            data_test.reset_index(drop=True),
        )

    def scale_pandas(self, dataframe, scaler):
        return pd.DataFrame(
            scaler.transform(dataframe.values), columns=dataframe.columns
        )

    def get_all_particles_dataset(self, dtype=None, log=False, n_quantiles=100000):
        data_train_all = []
        data_val_all = []
        scaler_all = {}
        for index, particle in enumerate(self.list_particles):
            data_train, data_val, scaler = self.get_merged_typed_dataset(
                particle, dtype=dtype, log=log, n_quantiles=n_quantiles
            )
            ohe_table = pd.DataFrame(
                np.zeros((len(data_train), len(self.list_particles))),
                columns=["is_{}".format(i) for i in self.list_particles],
            )
            ohe_table["is_{}".format(particle)] = 1

            data_train_all.append(
                pd.concat(
                    [
                        data_train.iloc[:, : self.y_count],
                        ohe_table,
                        data_train.iloc[:, self.y_count :],
                    ],
                    axis=1,
                )
            )

            data_val_all.append(
                pd.concat(
                    [
                        data_val.iloc[:, : self.y_count],
                        ohe_table[: len(data_val)].copy(),
                        data_val.iloc[:, self.y_count :],
                    ],
                    axis=1,
                )
            )
            scaler_all[index] = scaler
        data_train_all = pd.concat(data_train_all, axis=0).astype(dtype, copy=False)
        data_val_all = pd.concat(data_val_all, axis=0).astype(dtype, copy=False)
        return data_train_all, data_val_all, scaler_all

    def get_merged_typed_dataset(
        self, particle_type, dtype=None, log=False, n_quantiles=100000
    ):
        file_list = self.datasets[particle_type]
        if log:
            print("Reading and concatenating datasets:")
            for fname in file_list:
                print("\t{}".format(fname))
        data_full = self.load_and_merge_and_cut(file_list)
        # Must split the whole to preserve train/test split""
        if log:
            print("splitting to train/val/test")
        data_train, data_val, _ = self.split(data_full)
        if log:
            print("fitting the scaler")
        print("scaler train sample size: {}".format(len(data_train)))
        start_time = time()
        if n_quantiles == 0:
            scaler = StandardScaler().fit(
                data_train.drop(self.weight_col, axis=1).values
            )
        else:
            scaler = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=n_quantiles,
                subsample=int(1e10),
            ).fit(data_train.drop(self.weight_col, axis=1).values)
        print(
            "scaler n_quantiles: {}, time = {}".format(n_quantiles, time() - start_time)
        )
        if log:
            print("scaling train set")
        data_train = pd.concat(
            [
                self.scale_pandas(data_train.drop(self.weight_col, axis=1), scaler),
                data_train[self.weight_col],
            ],
            axis=1,
        )
        if log:
            print("scaling test set")
        data_val = pd.concat(
            [
                self.scale_pandas(data_val.drop(self.weight_col, axis=1), scaler),
                data_val[self.weight_col],
            ],
            axis=1,
        )
        if dtype is not None:
            if log:
                print("converting dtype to {}".format(dtype))
            data_train = data_train.astype(dtype, copy=False)
            data_val = data_val.astype(dtype, copy=False)
        return data_train, data_val, scaler


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
