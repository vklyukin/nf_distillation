import numpy as np
from typing import Tuple
import pandas as pd
from omegaconf.dictconfig import DictConfig


def get_ecdf(sample: np.array, weights: np.array = None) -> Tuple[np.array, np.array]:

    assert len(sample.shape) == 1, "Only 1D CDF is implemented"

    if weights is None:
        weights = np.ones_like(sample, dtype=sample.dtype)
    assert sample.shape == weights.shape

    i = np.argsort(sample)
    x, w = sample[i], weights[i]

    w_cumsum = np.cumsum(w)
    assert w_cumsum[-1] > 0

    w_cumsum /= w_cumsum[-1]
    return x, w_cumsum


def _interleave_ecdfs(
    x1: np.array, y1: np.array, x2: np.array, y2: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Interleave two eCDFs by their argument
    """
    assert len(x1.shape) == len(x2.shape) == 1
    assert x1.shape == y1.shape
    assert x2.shape == y2.shape

    x = np.sort(np.concatenate([x1, x2]))
    y1 = np.insert(y1, 0, [0])
    y2 = np.insert(y2, 0, [0])
    return (
        x,
        y1[np.searchsorted(x1, x, side="right")],
        y2[np.searchsorted(x2, x, side="right")],
    )


def ks_2samp_w(
    data1: np.array, data2: np.array, w1: np.array = None, w2: np.array = None
) -> float:
    cdf1 = get_ecdf(data1, w1)
    cdf2 = get_ecdf(data2, w2)
    _, cdf1_i, cdf2_i = _interleave_ecdfs(*cdf1, *cdf2)
    return np.abs(cdf2_i - cdf1_i).max()


def weighted_ks(
    config: DictConfig,
    data_real: pd.DataFrame,
    data_fake: pd.DataFrame,
    context: pd.DataFrame,
    weights: np.array,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_avg, results_max = pd.DataFrame(), pd.DataFrame()
    for feature_column in context.columns:
        bins = pd.qcut(context[feature_column], config.metric.ks.bins)
        for target_column in data_real.columns:
            df = pd.DataFrame(
                {
                    "real": data_real[target_column],
                    "fake": data_fake[target_column],
                    "weight": weights,
                    "bin": bins,
                }
            )
            group = df.groupby("bin")

            def calculate_ks(df):
                try:
                    return ks_2samp_w(
                        df["real"].to_numpy(),
                        df["fake"].to_numpy(),
                        df["weight"].to_numpy(),
                        df["weight"].to_numpy(),
                    )
                except AssertionError as e:
                    print("Assertion triggered:", e)
                    return np.nan

            ks_values = group.apply(calculate_ks)
            sizes = group["weight"].sum()

            selection = ~ks_values.isna()
            ks_values = ks_values[selection]
            sizes = sizes[selection]

            avg_ks = (ks_values * sizes).sum() / sizes.sum()
            max_ks = ks_values.max()

            results_avg.loc[(target_column, feature_column)] = avg_ks
            results_max.loc[(target_column, feature_column)] = max_ks

    return (
        results_avg,
        results_max,
    )
