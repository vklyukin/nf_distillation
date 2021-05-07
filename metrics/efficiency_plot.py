from collections.abc import Iterable
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, FixedLocator
import matplotlib.pyplot as plt
from omegaconf.dictconfig import DictConfig
from typing import Tuple, Generator


def make_efficiency_figure(
    config: DictConfig,
    real_column: pd.Series,
    fake_column: pd.Series,
    feature_column: pd.Series,
    weight_column: np.array,
    quantiles,
) -> Tuple[str, plt.Figure]:
    # quantiles = [float(q) for q in quantiles]
    bins = np.quantile(
        feature_column.to_numpy(),
        np.linspace(0.0, 1.0, config.metric.efficiency.bins + 1),
    )

    title = "{} efficiency{} vs {}{}".format(
        real_column.name,
        " ratio" if config.metric.efficiency.make_ratio else "",
        feature_column.name,
        f" at {quantiles}" if not isinstance(quantiles, Iterable) else "",
    )
    if not isinstance(quantiles, Iterable):
        quantiles = [quantiles]

    thresholds = np.quantile(real_column, quantiles)

    df = pd.DataFrame(
        {
            "real": real_column.values,
            "fake": fake_column.values,
            "feature": feature_column.values,
            "weight": weight_column,
        }
    )

    df["bin"] = pd.cut(df["feature"], bins=bins)
    group = df.groupby("bin")

    def calculate_efficiencies_or_their_ratios(df):
        total = df["weight"].sum()
        if total <= 0:
            result = pd.Series(
                [np.nan]
                * (len(quantiles) * (3 if config.metric.efficiency.make_ratio else 6))
            )
            if config.metric.efficiency.make_ratio:
                result.index = (
                    [f"eff_ratio_{q}" for q in quantiles]
                    + [f"eff_ratio_err_low_{q}" for q in quantiles]
                    + [f"eff_ratio_err_high_{q}" for q in quantiles]
                )
            else:
                result.index = (
                    sum(([f"eff_real_{q}", f"eff_fake_{q}"] for q in quantiles), [])
                    + sum(
                        (
                            [f"eff_real_err_low_{q}", f"eff_fake_err_low_{q}"]
                            for q in quantiles
                        ),
                        [],
                    )
                    + sum(
                        (
                            [f"eff_real_err_high_{q}", f"eff_fake_err_high_{q}"]
                            for q in quantiles
                        ),
                        [],
                    )
                )
            return result

        passed = pd.concat(
            [
                ((df[["real", "fake"]] >= thr) * df[["weight"]].to_numpy()).sum(axis=0)
                for thr in thresholds
            ],
            axis=0,
        ).clip(lower=0.0)
        if config.metric.efficiency.make_ratio:
            efficiencies = ((passed + 0.5) / (total + 1.0)).clip(0, 1)
        else:
            efficiencies = (passed / total).clip(0, 1)

        efficiencies.index = sum(
            ([f"eff_real_{q}", f"eff_fake_{q}"] for q in quantiles), []
        )

        if config.metric.efficiency.make_ratio:
            # Calculating ratio with 1-sigma confidence interval using the formula from here:
            # https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/binoraci.htm
            eff_real = efficiencies[[f"eff_real_{q}" for q in quantiles]].to_numpy()
            eff_fake = efficiencies[[f"eff_fake_{q}" for q in quantiles]].to_numpy()

            efficiency_ratios = eff_fake / eff_real
            exp_radical = np.exp(
                (
                    (1 - eff_real) / (total * eff_real)
                    + (1 - eff_fake) / (total * eff_fake)
                )
                ** 0.5
            )
            errors_low = efficiency_ratios - efficiency_ratios / exp_radical
            errors_high = efficiency_ratios * exp_radical - efficiency_ratios

            efficiency_ratios = pd.Series(
                efficiency_ratios, index=[f"eff_ratio_{q}" for q in quantiles]
            )
            errors_low = pd.Series(
                errors_low, index=[f"eff_ratio_err_low_{q}" for q in quantiles]
            )
            errors_high = pd.Series(
                errors_high, index=[f"eff_ratio_err_high_{q}" for q in quantiles]
            )
            result = pd.concat([efficiency_ratios, errors_low, errors_high], axis=0)

        else:
            # Calculating 1-sigma Wilson confidence interval as `mode +\- delta`
            mode = (efficiencies + 1.0 / (2.0 * total)) / (1.0 + 1.0 / total)
            delta = (
                efficiencies * (1 - efficiencies) / total + 1.0 / (4.0 * total ** 2)
            ).clip(lower=0) ** 0.5 / (1.0 + 1.0 / total)

            errors_low = efficiencies - (mode - delta)
            errors_high = (mode + delta) - efficiencies

            errors_low.index = sum(
                ([f"eff_real_err_low_{q}", f"eff_fake_err_low_{q}"] for q in quantiles),
                [],
            )
            errors_high.index = sum(
                (
                    [f"eff_real_err_high_{q}", f"eff_fake_err_high_{q}"]
                    for q in quantiles
                ),
                [],
            )

            result = pd.concat([efficiencies, errors_low, errors_high], axis=0)

        return result

    efficiencies = group.apply(calculate_efficiencies_or_their_ratios)

    figure = plt.figure(figsize=(8, 8))

    if config.metric.efficiency.make_ratio:
        plt.yscale("symlog", linthresh=1.0)
        plt.grid(b=True, which="major", linewidth=1.25)
        plt.grid(b=True, which="minor", linewidth=0.3)
        yaxis = plt.gca().yaxis
        yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x + 1}"))
        major_ticks = np.array(
            [-1.0, -0.5, 0.0, 0.5, 1.0, 9.0, 49.0, 99.0, 499.0, 999.0]
        )
        minor_ticks = np.concatenate(
            [
                np.linspace(l, r, 5 if i < 4 else (8 if i == 4 else 10), endpoint=False)
                for i, (l, r) in enumerate(zip(major_ticks[:-1], major_ticks[1:]))
            ],
            axis=0,
        )
        yaxis.set_major_locator(FixedLocator(major_ticks))
        yaxis.set_minor_locator(FixedLocator(minor_ticks))
    for q in quantiles:
        if config.metric.efficiency.make_ratio:
            args = {"fmt": "o", "marker": "o", "ms": 4, "markeredgewidth": 2}
            args["label"] = f'{q * 100}% {args.get("label", "")}'
            efficiencies.to_csv("./ttt.csv")
            plt.errorbar(
                x=efficiencies.index.mid,
                y=efficiencies[f"eff_ratio_{q}"] - 1.0,
                xerr=(efficiencies.index.right - efficiencies.index.left) / 2,
                yerr=efficiencies[
                    [f"eff_ratio_err_low_{q}", f"eff_ratio_err_high_{q}"]
                ].T.to_numpy(),
                **args,
            )
        else:
            real_args = {"fmt": "o", "marker": "o", "ms": 4, "markeredgewidth": 2}
            fake_args = {"fmt": "o", "marker": "o", "ms": 4, "markeredgewidth": 2}
            for args in [real_args, fake_args]:
                if len(quantiles) > 1:
                    args["label"] = f'{q * 100}% {args.get("label", "")}'

            plt.errorbar(
                x=efficiencies.index.mid,
                y=efficiencies[f"eff_real_{q}"],
                xerr=(efficiencies.index.right - efficiencies.index.left) / 2,
                yerr=efficiencies[
                    [f"eff_real_err_low_{q}", f"eff_real_err_high_{q}"]
                ].T.to_numpy(),
                **real_args,
            )
            plt.errorbar(
                x=efficiencies.index.mid,
                y=efficiencies[f"eff_fake_{q}"],
                xerr=(efficiencies.index.right - efficiencies.index.left) / 2,
                yerr=efficiencies[
                    [f"eff_fake_err_low_{q}", f"eff_fake_err_high_{q}"]
                ].T.to_numpy(),
                **fake_args,
            )
    if config.metric.efficiency.make_ratio:
        ymin, ymax = plt.gca().get_ylim()
        if ymin > -1.0:
            plt.ylim(bottom=-1.0)
        if ymax < 1.0:
            plt.ylim(top=1.0)
    if feature_column.name in ["Brunel_P", "nTracks_Brunel"]:
        plt.xscale("log")
    plt.legend()
    plt.title(title)

    return title, figure


def make_figures(
    config: DictConfig,
    features: pd.DataFrame,
    targets_real: pd.DataFrame,
    targets_fake: pd.DataFrame,
    weights: np.array,
):

    for target_column in targets_real.columns:
        for feature_column in features.columns:
            try:
                if config.metric.efficiency.make_ratio:
                    yield make_efficiency_figure(
                        config,
                        real_column=targets_real[target_column],
                        fake_column=targets_fake[target_column],
                        feature_column=features[feature_column],
                        weight_column=weights,
                        quantiles=config.metric.efficiency.thresholds,
                    )
                else:
                    for threshold in config.metric.efficiency.thresholds:
                        yield make_efficiency_figure(
                            config,
                            real_column=targets_real[target_column],
                            fake_column=targets_fake[target_column],
                            feature_column=features[feature_column],
                            weight_column=weights,
                            quantiles=threshold,
                        )
            except Exception as e:
                # print(e)
                # continue
                raise
