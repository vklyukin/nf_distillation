from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
import numpy as np
import pandas as pd
import os
from time import time

data_dir = "/data/vnkljukin/nf/data_calibsample/"


def get_particle_dset(particle):
    return [data_dir + name for name in os.listdir(data_dir) if particle in name]


list_particles = ["kaon", "pion", "proton", "muon", "electron"]
PARTICLES = list_particles

datasets = {particle: get_particle_dset(particle) for particle in list_particles}


dll_columns = ["RichDLLe", "RichDLLk", "RichDLLmu", "RichDLLp", "RichDLLbt"]
raw_feature_columns = ["Brunel_P", "Brunel_ETA", "nTracks_Brunel"]
weight_col = "probe_sWeight"

y_count = len(dll_columns)
TEST_SIZE = 0.5


def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter="\t")
    return data[dll_columns + raw_feature_columns + [weight_col]]


def load_and_merge_and_cut(filename_list):
    return pd.concat(
        [load_and_cut(os.path.join(os.getcwd(), fname)) for fname in filename_list],
        axis=0,
        ignore_index=True,
    )


def split(data):
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    data_val, data_test = train_test_split(
        data_val, test_size=TEST_SIZE, random_state=1812
    )
    return (
        data_train.reset_index(drop=True),
        data_val.reset_index(drop=True),
        data_test.reset_index(drop=True),
    )


def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)


def get_all_particles_dataset(dtype=None, log=False, n_quantiles=100000):
    data_train_all = []
    data_val_all = []
    scaler_all = {}
    for index, particle in enumerate(list_particles):
        data_train, data_val, scaler = get_merged_typed_dataset(
            particle, dtype=dtype, log=log, n_quantiles=n_quantiles
        )
        ohe_table = pd.DataFrame(
            np.zeros((len(data_train), len(list_particles))),
            columns=["is_{}".format(i) for i in list_particles],
        )
        ohe_table["is_{}".format(particle)] = 1

        data_train_all.append(
            pd.concat(
                [data_train.iloc[:, :y_count], ohe_table, data_train.iloc[:, y_count:]],
                axis=1,
            )
        )

        data_val_all.append(
            pd.concat(
                [
                    data_val.iloc[:, :y_count],
                    ohe_table[: len(data_val)].copy(),
                    data_val.iloc[:, y_count:],
                ],
                axis=1,
            )
        )
        scaler_all[index] = scaler
    data_train_all = pd.concat(data_train_all, axis=0).astype(dtype, copy=False)
    data_val_all = pd.concat(data_val_all, axis=0).astype(dtype, copy=False)
    return data_train_all, data_val_all, scaler_all


def get_merged_typed_dataset(particle_type, dtype=None, log=False, n_quantiles=100000):
    file_list = datasets[particle_type]
    if log:
        print("Reading and concatenating datasets:")
        for fname in file_list:
            print("\t{}".format(fname))
    data_full = load_and_merge_and_cut(file_list)
    # Must split the whole to preserve train/test split""
    if log:
        print("splitting to train/val/test")
    data_train, data_val, _ = split(data_full)
    if log:
        print("fitting the scaler")
    print("scaler train sample size: {}".format(len(data_train)))
    start_time = time()
    if n_quantiles == 0:
        scaler = StandardScaler().fit(data_train.drop(weight_col, axis=1).values)
    else:
        scaler = QuantileTransformer(
            output_distribution="normal", n_quantiles=n_quantiles, subsample=int(1e10)
        ).fit(data_train.drop(weight_col, axis=1).values)
    print("scaler n_quantiles: {}, time = {}".format(n_quantiles, time() - start_time))
    if log:
        print("scaling train set")
    data_train = pd.concat(
        [
            scale_pandas(data_train.drop(weight_col, axis=1), scaler),
            data_train[weight_col],
        ],
        axis=1,
    )
    if log:
        print("scaling test set")
    data_val = pd.concat(
        [scale_pandas(data_val.drop(weight_col, axis=1), scaler), data_val[weight_col]],
        axis=1,
    )
    if dtype is not None:
        if log:
            print("converting dtype to {}".format(dtype))
        data_train = data_train.astype(dtype, copy=False)
        data_val = data_val.astype(dtype, copy=False)
    return data_train, data_val, scaler
