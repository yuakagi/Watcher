"""Utils for data visualization"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from .paletes import CUSTOM_PALETTES


def cm_to_inch(cm: float):
    """Converts centimeter to inch."""
    return cm / 2.54


def save_as_svg(fig, path):
    """Save fig as SVG, the best format for scientific figs."""
    if not path.endswith(".svg"):
        path += ".svg"
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    print(f"Figure saved at {path}")


def set_figure_params(
    font_family="Arial",
    font_small=5,
    font_medium=6,
    font_large=7,
    spine_width=0.5,
    line_width=1,
    tick_width=0.5,
    tick_size=1.5,
    marker_size=1.0,
    dpi=700,
):
    """Configure matplotlib parameters for consistent figure styling.

    Args:
        font_family (str): Font family to use.
        font_small (float): General font size.
        font_medium (float): Axis label font size.
        font_large (float): Title font size.
        spine_width (float): Line width of axes spines.
        line_width (float): Default line width.
        tick_width (float): Width of major/minor ticks.
        tick_size (float): Length of major/minor ticks.
        marker_size (float): Marker size.
        dpi (int): Resolution of figures.
        **kwargs: Additional rcParams to override.
    """
    # Font
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = font_small
    plt.rcParams["axes.labelsize"] = font_medium
    plt.rcParams["axes.titlesize"] = font_large
    plt.rcParams["axes.titlepad"] = font_large / 2
    plt.rcParams["axes.labelpad"] = font_medium / 2
    # Lines
    plt.rcParams["axes.linewidth"] = spine_width
    plt.rcParams["lines.linewidth"] = line_width
    # Markers
    plt.rcParams["lines.markersize"] = marker_size
    # Ticks
    plt.rcParams["ytick.major.width"] = tick_width
    plt.rcParams["ytick.minor.width"] = tick_width
    plt.rcParams["xtick.major.width"] = tick_width
    plt.rcParams["xtick.minor.width"] = tick_width
    plt.rcParams["ytick.major.size"] = tick_size
    plt.rcParams["ytick.minor.size"] = tick_size
    plt.rcParams["xtick.major.size"] = tick_size
    plt.rcParams["xtick.minor.size"] = tick_size
    plt.rcParams["xtick.major.pad"] = tick_size / 2
    plt.rcParams["ytick.major.pad"] = tick_size / 2
    plt.rcParams["xtick.minor.pad"] = tick_size / 2
    plt.rcParams["ytick.minor.pad"] = tick_size / 2
    # Figure
    plt.rcParams["figure.dpi"] = dpi

    return plt.rcParams


def _roc(y: np.ndarray, t: np.ndarray, thresholds: np.ndarray, epsilon: float = 1e-8):
    # Reshape for broadcasting
    preds = thresholds.reshape(-1, 1) <= y.reshape(1, -1)  # (n_thresholds, n_samples)
    t_bool = t.astype(bool).reshape(1, -1)  # shape: (1, n_samples)
    # Compute confusion matrix elements
    tp = np.sum(preds & t_bool, axis=1)  # shape: (n_thresholds,)
    fn = np.sum(~preds & t_bool, axis=1)
    tn = np.sum(~preds & ~t_bool, axis=1)
    fp = np.sum(preds & ~t_bool, axis=1)
    # Compute TPR and FPR
    tpr = np.divide(tp, tp + fn + epsilon)
    fpr = np.divide(fp, fp + tn + epsilon)
    # Compute auc
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    df = df.sort_values("fpr")
    df["base"] = (df["fpr"] - df["fpr"].shift(1)).fillna(0)
    df["shifted tpr"] = df["tpr"].shift(1, fill_value=0)
    df["higher height"] = df[["tpr", "shifted tpr"]].max(axis=1)
    df["lower height"] = df[["tpr", "shifted tpr"]].min(axis=1)
    auc = (
        df["base"] * df["lower height"]
        + 0.5 * (df["base"] * (df["higher height"] - df["lower height"]))
    ).sum()
    # Finalize
    fpr = df["fpr"].tolist()
    tpr = df["tpr"].tolist()
    return fpr, tpr, auc


def _boot_roc(y: np.ndarray, t: np.ndarray, thresholds: np.ndarray):
    """Computes ROC and AUROC by non-parametric bootstrapping."""
    n_items = y.shape[0]
    indexes = np.arange(0, n_items)
    # Non-parametric bootstrapping
    resampled_idx = np.random.choice(indexes, size=n_items, replace=True)
    resampled_y = y[resampled_idx]
    resampled_t = t[resampled_idx]
    sorted_new_idx = resampled_y.argsort()
    resampled_y = resampled_y[sorted_new_idx]
    resampled_t = resampled_t[sorted_new_idx]
    _, _, boot_auc = _roc(resampled_y, resampled_t, thresholds)
    return boot_auc


def roc(
    y: list | np.ndarray,
    t: list | np.ndarray,
    n_thresholds: int = 100,
    percentile: int = 95,
    n_resampling: int = 100,
    max_workers: int = 1,
    epsilon: float = 1e-8,
):
    """Computes ROC and AUROC."""
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    sorted_index = y.argsort()
    y = y[sorted_index]
    t = t[sorted_index]
    thresholds = np.linspace(y.min() - 1e-8, y.max() + 1e-8, n_thresholds)

    # Compute the raw auc value
    raw_fpr, raw_tpr, raw_auc = _roc(y, t, thresholds)

    # Compute confidence intervals by resampling
    if percentile:
        lower_p = (100 - percentile) / 2
        upper_p = 100 - lower_p
        boot_auc_list = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_boot_roc, y, t, thresholds)
                for _ in range(n_resampling)
            ]
            for future in tqdm(
                as_completed(futures), desc="Bootstrapping ROC", total=n_resampling
            ):
                boot_auc = future.result()
                boot_auc_list.append(boot_auc)

        # Finalize results
        boot_aucs = np.array(boot_auc_list)
        auc_low = np.percentile(boot_aucs, lower_p)
        auc_high = np.percentile(boot_aucs, upper_p)
        auc = (raw_auc, auc_low, auc_high)

    else:
        auc = (raw_auc, None, None)

    # Best thresholds by F1 maximization
    # Compute confusion matrix for all thresholds
    predicted_pos = y >= thresholds.reshape(-1, 1)  # 2D (thresholds x observations)
    predicted_neg = np.logical_not(predicted_pos)  # 2D (thresholds x observations)
    positives = t.astype(bool)  # 1D
    negatives = np.logical_not(positives)  # 1D
    tp = np.sum(predicted_pos & positives, axis=1)  # 1D (thresholds, )
    fp = np.sum(predicted_pos & negatives, axis=1)
    fn = np.sum(predicted_neg & positives, axis=1)
    tn = np.sum(predicted_neg & negatives, axis=1)
    sensitivity = recall = tp / np.maximum(tp + fn, epsilon)  # 1D (thresholds, )
    specificity = tn / np.maximum(tn + fp, epsilon)
    precision = tp / np.maximum(tp + fp, epsilon)
    negative_predictive_value = tn / np.maximum(tn + fn, epsilon)
    accuracy = (tp + tn) / np.maximum(tp + tn + fp + fn, epsilon)
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, epsilon)
    # Determine the best thresholds (Use raw) by ROC distance
    distance = np.sqrt((1 - specificity) ** 2 + (1 - sensitivity) ** 2)
    best_roc_idx = np.argmin(distance)
    # Determine the bes
    best_f1_idx = np.argmax(f1)
    stats = {
        "best_by_ROC": {
            "best_threshold": thresholds[best_roc_idx],
            "sensitivity": sensitivity[best_roc_idx],
            "specificity": specificity[best_roc_idx],
            "precision": precision[best_roc_idx],
            "negative_predictive_value": negative_predictive_value[best_roc_idx],
            "accuracy": accuracy[best_roc_idx],
            "f1": f1[best_roc_idx],
        },
        "best_by_F1": {
            "best_threshold": thresholds[best_f1_idx],
            "sensitivity": sensitivity[best_f1_idx],
            "specificity": specificity[best_f1_idx],
            "precision": precision[best_f1_idx],
            "negative_predictive_value": negative_predictive_value[best_f1_idx],
            "accuracy": accuracy[best_f1_idx],
            "f1": f1[best_f1_idx],
        },
        "all": {
            "thresholds": thresholds,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "precision": precision,
            "negative_predictive_value": negative_predictive_value,
            "accuracy": accuracy,
            "f1": f1,
        },
    }

    return raw_fpr, raw_tpr, auc, stats


def plot_roc(
    ys: list[np.ndarray] | np.ndarray,
    ts: list[np.ndarray] | np.ndarray,
    labels: list[str],
    ax,
    n_thresholds: int = 100,
    percentile: int = 95,
    n_resampling: int = 100,
    colors: list[str] | None = None,
    alphas: list[float] | None = None,
    verbose: bool = True,
):
    """Plots ROC with AUROC, computing 95%CI by non-parametric bootstrapping."""
    # Check args
    if isinstance(ys, np.ndarray):
        ys = [ys]
    if isinstance(ts, np.ndarray):
        ts = [ts]
    # Line for random guess
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=plt.rcParams["axes.linewidth"],
    )

    # Generate colors
    if colors is None:
        num_lines = len(ys) + 2
        colors = plt.cm.plasma(np.linspace(1, 0, num_lines))
        colors = colors[1:-1]
    if alphas is None:
        alphas = [1 for _ in range(len(colors))]

    # Plot lines
    final_stats = {}
    for y, t, label, color, alpha in zip(ys, ts, labels, colors, alphas):
        raw_fpr, raw_tpr, auc, stats = roc(
            y=y,
            t=t,
            n_thresholds=n_thresholds,
            percentile=percentile,
            n_resampling=n_resampling,
        )

        # AUC
        a_raw, a_low, a_high = auc
        if (a_low is not None) and (a_high is not None):
            metrics_str = f"{a_raw:.3f} ({a_low:.3f} – {a_high:.3f})"
        else:
            metrics_str = f"{a_raw:.3f}"

        # ROC
        label_auc = f"{label}: {metrics_str}"
        sns.lineplot(
            x=raw_fpr,
            y=raw_tpr,
            ax=ax,
            color=color,
            label=label_auc,
            alpha=alpha,
            errorbar=None,
        )

        # Verbose
        # Print confusion matrix vals and thresholds
        if verbose:
            print(f"=== {label}: stats ===")
            for k1, v1 in stats.items():
                if k1.startswith("best"):
                    print(" ".join(k1.split("_")))
                    for k2, v2 in v1.items():
                        print("--", " ".join(k2.split("_")), ":", v2)

        # Save for final stats
        final_stats[label] = stats["all"]

    # Configure ax and legend
    legend = ax.legend(
        loc="lower right",
        title=f"AUROC with {percentile}%CI",
        handletextpad=0.5,
        borderaxespad=0.3,
        labelspacing=0.2,
        fancybox=False,
        frameon=True,
        edgecolor="none",
    )
    legend.get_frame().set_facecolor(CUSTOM_PALETTES["bg_gray"][0])
    legend.get_frame().set_alpha(0.4)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    return ax, final_stats


def _pr(
    y: np.ndarray, t: np.ndarray, thresholds: np.ndarray
) -> tuple[list, list, float]:
    precisions, recalls = [], []
    preds = thresholds.reshape(-1, 1) <= y.reshape(1, -1)  # (n_thresholds, n_samples)
    t_bool = t.astype(bool).reshape(1, -1)  # (1, n_samples)
    tp = np.sum(preds & t_bool, axis=1)  # sum across samples
    fn = np.sum(~preds & t_bool, axis=1)
    fp = np.sum(preds & ~t_bool, axis=1)
    denom_p = tp + fp
    denom_r = tp + fn
    precision = np.divide(
        tp, denom_p, out=np.full_like(tp, np.nan, dtype=float), where=denom_p > 0
    )
    recall = np.divide(
        tp, denom_r, out=np.full_like(tp, np.nan, dtype=float), where=denom_r > 0
    )
    precisions = precision.tolist()
    recalls = recall.tolist()

    # Compute auc
    df = pd.DataFrame({"rc": recalls, "pr": precisions})
    df = df.dropna(
        subset=["rc", "pr"]
    )  # Drop any threshold where either precision or recall is NaN
    df = df.sort_values("rc")
    df["base"] = (df["rc"] - df["rc"].shift(1)).fillna(0)
    df["shifted pr"] = df["pr"].shift(1, fill_value=0)
    df["higher height"] = df[["pr", "shifted pr"]].max(axis=1)
    df["lower height"] = df[["pr", "shifted pr"]].min(axis=1)
    auc = (
        df["base"] * df["lower height"]
        + 0.5 * (df["base"] * (df["higher height"] - df["lower height"]))
    ).sum()
    # If the recall doesn't start from 0, add the area of the rectangle from recall=0 to the first recall value
    if df["rc"].min() > 0:
        auc += (
            df["rc"].min() * 1.0
        )  # Assuming precision = 1 when recall < first recall point
    # Finalize
    recalls = df["rc"].tolist()
    precisions = df["pr"].tolist()

    return recalls, precisions, auc


def _boot_pr(y: np.ndarray, t: np.ndarray, thresholds: np.ndarray) -> float:
    n_items = y.shape[0]
    indexes = np.arange(0, n_items)
    # Non-parametric bootstrapping
    resampled_idx = np.random.choice(indexes, size=n_items, replace=True)
    resampled_y = y[resampled_idx]
    resampled_t = t[resampled_idx]
    sorted_new_idx = resampled_y.argsort()
    resampled_y = resampled_y[sorted_new_idx]
    resampled_t = resampled_t[sorted_new_idx]
    _, _, boot_auc = _pr(resampled_y, resampled_t, thresholds)
    return boot_auc


def pr(
    y: list | np.ndarray,
    t: list | np.ndarray,
    n_thresholds: int = 100,
    percentile: int = 95,
    n_resampling: int = 100,
    max_workers: int = 1,
):
    """Computes PR curve."""
    if n_thresholds is None:
        n_thresholds = 100
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    sorted_index = y.argsort()
    y = y[sorted_index]
    t = t[sorted_index]
    # Thresholds to cover the whole range of y
    thresholds = np.linspace(y.min() - 1e-8, y.max() + 1e-8, n_thresholds)
    # PR curve
    raw_recalls, raw_precisions, raw_auc = _pr(y, t, thresholds)

    # Compute confidence intervals by resampling
    if percentile:
        lower_p = (100 - percentile) / 2
        upper_p = 100 - lower_p
        boot_auc_list = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_boot_pr, y, t, thresholds) for _ in range(n_resampling)
            ]
            for future in tqdm(
                as_completed(futures), desc="Bootstrapping PR", total=n_resampling
            ):
                boot_auc = future.result()
                boot_auc_list.append(boot_auc)
        # Finalize results
        boot_aucs = np.array(boot_auc_list)
        auc_low = np.percentile(boot_aucs, lower_p)
        auc_high = np.percentile(boot_aucs, upper_p)
        auc = (raw_auc, auc_low, auc_high)

    else:
        auc = (raw_auc, None, None)

    return raw_recalls, raw_precisions, auc


def plot_prc(
    ys: list[np.ndarray] | np.ndarray,
    ts: list[np.ndarray] | np.ndarray,
    labels: list[str],
    ax,
    n_thresholds: int = 100,
    percentile: int = 95,
    n_resampling: int = 100,
    colors: list[str] | None = None,
    alphas: list[float] | None = None,
    verbose: bool = True,
):
    """Plots PR with AUPRC, computing 95%CI by non-parametric bootstrapping."""
    # Check args
    if isinstance(ys, np.ndarray):
        ys = [ys]
    if isinstance(ts, np.ndarray):
        ts = [ts]

    # Generate colors
    if colors is None:
        num_lines = len(ys) + 2
        colors = plt.cm.plasma(np.linspace(1, 0, num_lines))
        colors = colors[1:-1]
    if alphas is None:
        alphas = [1 for _ in range(len(colors))]

    # Plot lines
    for y, t, label, color, alpha in zip(ys, ts, labels, colors, alphas):
        raw_recalls, raw_precisions, auc = pr(
            y=y,
            t=t,
            n_thresholds=n_thresholds,
            percentile=percentile,
            n_resampling=n_resampling,
        )

        # AUC
        a_raw, a_low, a_high = auc
        if (a_low is not None) and (a_high is not None):
            metrics_str = f"{a_raw:.3f} ({a_low:.3f} – {a_high:.3f})"
        else:
            metrics_str = f"{a_raw:.3f}"

        # PRC
        label_auc = f"{label}: {metrics_str}"
        sns.lineplot(
            x=raw_recalls,
            y=raw_precisions,
            ax=ax,
            color=color,
            label=label_auc,
            alpha=alpha,
            errorbar=None,
        )

        # Baseline
        baseline_prec = raw_precisions[-1]
        ax.hlines(
            y=baseline_prec,
            xmax=1,
            xmin=0,
            color=color,
            linestyles="--",
            linewidth=plt.rcParams["axes.linewidth"],
            alpha=alpha,
        )
        if verbose:
            # Print relative AUPRC gain from the random baseline
            relative_auprc_gain = f"{a_raw/baseline_prec:.3f} ({a_low/baseline_prec:.3f} – {a_high/baseline_prec:.3f})"
            print("=====", label, "=====")
            print(
                f"N positive: {int(np.array(t).sum())}, N inference: {int(len(t))}, Baseline precision: {baseline_prec}"
            )
            print("Relative AUPRC gain from the random baseline", relative_auprc_gain)
            print("====================")

    # Configure ax and legend
    legend = ax.legend(
        loc="best",
        title=f"AUPRC with {percentile}%CI",
        handletextpad=0.5,
        borderaxespad=0.3,
        labelspacing=0.2,
        fancybox=False,
        frameon=True,
        edgecolor="none",
    )
    legend.get_frame().set_facecolor(CUSTOM_PALETTES["bg_gray"][0])
    legend.get_frame().set_alpha(0.4)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    return ax


def _compute_calib_bins(
    y: np.ndarray,
    t: np.ndarray,
    bin_lower_lims: np.ndarray,
    bin_center_vals: np.ndarray,
    bin_upper_lims: np.ndarray,
):
    # Initialize
    df = pd.DataFrame({"proba": y, "label": t})
    df["label"] = df["label"].astype(int)
    df["center"] = 0.0
    n_bins = bin_center_vals.shape[0]
    # Checking bins
    for i in range(n_bins):
        low = bin_lower_lims[i]
        high = bin_upper_lims[i]
        c = bin_center_vals[i]
        mask = df["proba"].between(
            low, high, inclusive="both" if i == n_bins - 1 else "left"
        )
        df.loc[mask, "center"] = c

    # NOTE:center_and_mean is indexed by center values, and values are mean values.
    center_and_mean = (
        df.groupby("center")["label"].mean().reindex(bin_center_vals, fill_value=np.nan)
    )

    pred_proba = center_and_mean.index.to_numpy()
    true_freq = center_and_mean.values

    return pred_proba, true_freq


def _quantify_calibration(y: np.ndarray, t: np.ndarray):
    """Quantifies calibration metrics for a given set of predicted probabilities and observed outcomes.
    Args:
        y (np.ndarray): Predicted probabilities (values between 0 and 1).
        t (np.ndarray): Observed binary outcomes (0 or 1).
    Returns:
        tuple: A tuple containing:
            - o_e_ratio (float): O/E ratio.
            - calibration_in_the_large (float): Calibration-in-the-large.
            - slope (float): Calibration slope.
            - intercept (float): Calibration intercept.
    """
    # Transform y to logit scale safely
    eps = 1e-10
    y_safe = np.clip(y, eps, 1 - eps)
    y_logit = np.log(y_safe / (1 - y_safe))
    try:
        # Compute CITL
        X = np.ones_like(t)
        logit_model = sm.Logit(t, X, offset=y_logit)
        result = logit_model.fit(disp=False)
        calibration_in_the_large = result.params[0]
        # Compute slope, intercept
        X = sm.add_constant(y_logit)
        logit_model = sm.Logit(t, X)
        result = logit_model.fit(disp=False)
    except Exception as e:
        print(" ****************** ")
        df = pd.DataFrame({"y_logit": y_logit, "t": t})
        print(pd.crosstab(df["y_logit"], df["t"]))
        print("Unique values in y_logit:", np.unique(y_logit))
        print("Standard deviation of y_logit:", np.std(y_logit))
        print("*********************")
        print(e)
        raise RuntimeError from e
    intercept = result.params[0]
    slope = result.params[1]
    # Compute O/E
    o_e_ratio = t.sum() / y.sum()
    return o_e_ratio, calibration_in_the_large, slope, intercept


def _boot_calib(
    loess_xs: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    bin_center_vals: np.ndarray,
    bin_lower_lims: np.ndarray,
    bin_upper_lims: np.ndarray,
    loess_kw: dict,
):
    n_items = y.shape[0]
    indexes = np.arange(0, n_items)
    # Non-parametric bootstrapping
    resampled_idx = np.random.choice(indexes, size=n_items, replace=True)
    resampled_y = y[resampled_idx]
    resampled_t = t[resampled_idx]
    sorted_new_idx = resampled_y.argsort()
    resampled_y = resampled_y[sorted_new_idx]
    resampled_t = resampled_t[sorted_new_idx]
    # Apply extremely small values to y in order to keep the array increasing
    resampled_y = resampled_y + np.linspace(0, 1e-10, n_items)
    # Loess
    if np.allclose(resampled_y, resampled_y[0]):
        # loess_ys = np.full_like(loess_xs, np.nan)
        loess_ys = np.full_like(loess_xs, resampled_t.mean())
        print(
            f"[Warning] Constant resampled_y = {resampled_y[0]:.4f}. Using prevalence = {resampled_t.mean():.4f}"
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            loess_ys = lowess(endog=resampled_t, exog=resampled_y, **loess_kw)
            if np.isnan(loess_ys).any():
                print(
                    "LOESS failed partially. Fraction of NaNs:",
                    np.isnan(loess_ys).mean(),
                )
    # Supplementary bins
    _, calib_y = _compute_calib_bins(
        resampled_y, resampled_t, bin_lower_lims, bin_center_vals, bin_upper_lims
    )
    # Quantify
    o_e_ratio, calibration_in_the_large, slope, intercept = _quantify_calibration(
        resampled_y, resampled_t
    )
    return loess_ys, calib_y, o_e_ratio, calibration_in_the_large, slope, intercept


def calib(
    y: np.ndarray,
    t: np.ndarray,
    n_bins: int = 10,
    loess_frac: float = 0.5,
    percentile: int = 95,
    n_resampling: int = 100,
    max_workers: int = 1,
):
    """Computes calibration curve"""

    # Initialize
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    # Clip once, immediately, and use this everywhere
    eps = 1e-4
    y = np.clip(y, eps, 1 - eps)
    # Sort
    sorted_index = y.argsort()
    y = y[sorted_index]
    t = t[sorted_index]
    # Remove NaNs and Infs
    valid_mask = ~np.isnan(y) & ~np.isnan(t) & ~np.isinf(y) & ~np.isinf(t)
    y_clean, t_clean = y[valid_mask], t[valid_mask]
    # Count items
    loess_xs = np.arange(y_clean.min(), y_clean.max(), 0.005)
    interval = 1 / n_bins
    bin_lower_lims = np.arange(0, 1, interval)
    bin_center_vals = bin_lower_lims + interval / 2
    bin_upper_lims = bin_lower_lims + interval
    quant_idx = ["oe", "citl", "slope", "intercept"]
    loess_df = pd.DataFrame(index=loess_xs)
    bins_df = pd.DataFrame(index=bin_center_vals)
    quant_df = pd.DataFrame(index=quant_idx, dtype=float)
    loess_kw = dict(frac=loess_frac, it=0, xvals=loess_xs)
    # Compute raw values
    raw_calib_xs, raw_calib_ys = _compute_calib_bins(
        y_clean, t_clean, bin_lower_lims, bin_center_vals, bin_upper_lims
    )
    # Apply LOWESS safely
    if np.allclose(y_clean, y_clean[0]):
        # raw_loess_ys = np.full_like(loess_xs, np.nan)
        raw_loess_ys = np.full_like(loess_xs, t_clean.mean())
        print(
            f"[Warning] Constant resampled_y = {y_clean[0]:.4f}. Using prevalence = {t_clean.mean():.4f}"
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw_loess_ys = lowess(endog=t_clean, exog=y_clean, **loess_kw)
    o_e_ratio, calibration_in_the_large, slope, intercept = _quantify_calibration(
        y_clean, t_clean
    )
    bins_df.loc[raw_calib_xs, "raw"] = raw_calib_ys
    loess_df.loc[loess_xs, "raw"] = raw_loess_ys
    quant_df.loc[quant_idx, "raw"] = [
        o_e_ratio,
        calibration_in_the_large,
        slope,
        intercept,
    ]

    # Resample
    loess_array = np.zeros((loess_xs.shape[0], n_resampling))
    calib_array = np.zeros((n_bins, n_resampling))
    quant_array = np.zeros((len(quant_idx), n_resampling))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _boot_calib,
                loess_xs,
                y_clean,
                t_clean,
                bin_center_vals,
                bin_lower_lims,
                bin_upper_lims,
                loess_kw,
            )
            for _ in range(n_resampling)
        ]
        for i, future in tqdm(
            enumerate(as_completed(futures)),
            desc="Bootstrapping calibration",
            total=n_resampling,
        ):
            loess_ys, calib_y, o_e_ratio, calibration_in_the_large, slope, intercept = (
                future.result()
            )
            loess_array[:, i] = loess_ys
            calib_array[:, i] = calib_y
            quant_array[:, i] = np.array(
                [o_e_ratio, calibration_in_the_large, slope, intercept]
            )

    # Stats
    lower_p = (100 - percentile) / 2
    upper_p = 100 - lower_p
    loess_df["low"] = np.nanpercentile(loess_array, q=lower_p, axis=1)
    loess_df["high"] = np.nanpercentile(loess_array, q=upper_p, axis=1)
    bins_df["low"] = np.nan
    bins_df["high"] = np.nan
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    hist_counts, _ = np.histogram(y_clean, bins=bin_boundaries)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_width = bin_boundaries[1] - bin_boundaries[0]
    bins_df["counts"] = hist_counts
    bins_df["bin_centers"] = bin_centers
    bins_df["bin_width"] = bin_width

    if not np.isnan(calib_array).all():
        valid_rows = ~np.isnan(calib_array).all(axis=1)
        bins_df.loc[valid_rows, "low"] = np.nanpercentile(
            calib_array[valid_rows], q=lower_p, axis=1
        )
        bins_df.loc[valid_rows, "high"] = np.nanpercentile(
            calib_array[valid_rows], q=upper_p, axis=1
        )
    quant_df["low"] = np.nan
    quant_df["high"] = np.nan
    if not np.isnan(quant_array).all():
        valid_rows = ~np.isnan(quant_array).all(axis=1)
        quant_df.loc[valid_rows, "low"] = np.nanpercentile(
            quant_array[valid_rows], q=lower_p, axis=1
        )
        quant_df.loc[valid_rows, "high"] = np.nanpercentile(
            quant_array[valid_rows], q=upper_p, axis=1
        )

    return loess_df, bins_df, quant_df


def plot_one_calib(
    loess: pd.DataFrame,
    bins: pd.DataFrame,
    quants: pd.DataFrame,
    label: str,
    ax,
    color,
    show_bars: bool = True,
    show_hist: bool = True,
    show_tick_labels=True,
    hist_log_scale: bool = True,
    plot_diagonal: bool = True,
    verbose: bool = True,
):
    """Plots one calibration using data from calib()"""
    if plot_diagonal:
        # Plot the ideal line
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            linewidth=plt.rcParams["axes.linewidth"],
        )
    # Ax for hist
    if show_hist:
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("bottom", size=0.15, pad=0.15, sharex=ax)
    else:
        ax_hist = None
        # Metrics
    oe_raw = quants.loc["oe", "raw"].item()
    oe_l = quants.loc["oe", "low"].item()
    oe_h = quants.loc["oe", "high"].item()
    oe_mid = (oe_l + oe_h) / 2
    oe_span = (oe_h - oe_l) / 2
    label_with_oe = label + f": {oe_mid:.3f} ± {oe_span:.3f}"
    if verbose:
        metrics = []
        # Sope
        slope_raw = quants.loc["slope", "raw"].item()
        slope_l = quants.loc["slope", "low"].item()
        slope_h = quants.loc["slope", "high"].item()
        metrics.append(f"Slope:{slope_raw:.3f} ({slope_l:.3f} – {slope_h:.3f})")
        # Intercept
        intercept_raw = quants.loc["intercept", "raw"].item()
        intercept_l = quants.loc["intercept", "low"].item()
        intercept_h = quants.loc["intercept", "high"].item()
        metrics.append(
            f"Intercept:{intercept_raw:.3f} ({intercept_l:.3f} – {intercept_h:.3f})"
        )
        # OE
        metrics.append(f"O/E:{oe_raw:.3f} ({oe_l:.3f} – {oe_h:.3f})")
        citl_raw = quants.loc["citl", "raw"].item()
        citl_l = quants.loc["citl", "low"].item()
        citl_h = quants.loc["citl", "high"].item()
        # Calibration-in-the-large
        metrics.append(
            f"Calibration-in-the-large:{citl_raw:.3f} ({citl_l:.3f} – {citl_h:.3f})"
        )
        joined_metrics = ",\n".join(metrics)
        metrics_str = f"***Metrics from '{label}'***\n {joined_metrics}"
        print(metrics_str)

    # Calibration curve
    sns.lineplot(
        x=loess.index,
        y=loess["raw"],
        ax=ax,
        label=label_with_oe,
        color=color,
    )  # Plot LOESS
    ax.fill_between(
        loess.index,
        loess["low"],
        loess["high"],
        color=color,
        alpha=0.3,
        edgecolor=None,
    )

    # Plot bins and errorbars
    if show_bars:
        # Skip bins with no items
        nonnull_bins = bins.dropna()
        y_error = np.abs(
            np.vstack([nonnull_bins["low"].values, nonnull_bins["high"].values])
            - nonnull_bins["raw"].values
        )
        ax.errorbar(
            x=nonnull_bins.index.values,
            y=nonnull_bins["raw"].values,
            yerr=y_error,
            capsize=plt.rcParams["lines.markersize"] * 1.2,
            ms=plt.rcParams["lines.markersize"],
            capthick=0.5,
            fmt="o",
            color=CUSTOM_PALETTES["bg_gray"][-1],
            linewidth=plt.rcParams["axes.linewidth"],
        )

    # Hist
    if show_hist:
        bin_centers = bins["bin_centers"]
        bin_width = bins["bin_width"]
        hist_counts = bins["counts"]
        if hist_log_scale:
            hist_counts = np.log10(hist_counts)
        ax_hist.bar(
            bin_centers,
            hist_counts,
            color=color,
            edgecolor="w",
            width=bin_width,
        )
        # Configure ax for hist
        ax_hist.set_xlabel(None)
        ax_hist.set_ylabel(None)
        ax_hist.spines[["right", "top"]].set_visible(False)
        # Refresh the ticks first
        if hist_log_scale:
            max_power = int(np.floor(hist_counts.max()))
            yticks = [p for p in range(max_power + 1)]
            yticklabels = []
            for t in yticks:
                if t == 0:
                    yticklabels.append("0")
                elif t == max_power:
                    yticklabels.append(r"$10^{{{}}}$".format(t))
                else:
                    yticklabels.append("")
        else:
            yticks = ax_hist.get_yticks()
            yticks[0] = 0
            yticklabels = ["" for _ in yticks]
            yticklabels[0] = 0
            yticklabels[-1] = int(yticks[-1])
        ax_hist.set_yticks(yticks)
        ax_hist.set_yticklabels(yticklabels)
        if not show_tick_labels:
            ax_hist.set_yticklabels([])
            ax_hist.set_xticklabels([])

    # Tick settings
    if not show_tick_labels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Spine and label settings
    ax.set_aspect(1)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.spines[["right", "top"]].set_visible(False)
    # Axis labels
    if show_hist:
        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("Count")
        ax.set_xlabel(None)
    else:
        ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")

    return ax, ax_hist


def plot_calibration(
    ys: list[np.ndarray] | np.ndarray,
    ts: list[np.ndarray] | np.ndarray,
    labels: list[str],
    ax,
    plot_last_only: bool = False,
    colors: list | None = None,
    verbose: bool = True,
    show_bars: bool = True,
    show_hist: bool = True,
    show_tick_labels=True,
    hist_log_scale: bool = True,
    n_bins: int = 10,
    loess_frac: float = 0.5,
    percentile: int = 95,
    n_resampling: int = 1000,
):
    """Plots calibration curve with metrics"""

    # Check args
    if isinstance(ys, np.ndarray):
        ys = [ys]
    if isinstance(ts, np.ndarray):
        ts = [ts]

    # Plot the ideal line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=plt.rcParams["axes.linewidth"],
    )

    # Ax for hist
    if show_hist:
        divider = make_axes_locatable(ax)
        ax_hist = divider.append_axes("bottom", size=0.15, pad=0.15, sharex=ax)
    else:
        ax_hist = None

    # Generate colors
    if colors is None:
        num_lines = len(ys) + 2
        # pylint: disable=no-member
        colors = plt.cm.plasma(np.linspace(1, 0, num_lines))
        colors = colors[1:-1]

    for i, (y, t, label, color) in enumerate(zip(ys, ts, labels, colors)):
        if not (plot_last_only and i != len(ys) - 1):
            # Compute plotted values and metrics
            loess, bins, quants = calib(
                y,
                t,
                n_bins=n_bins,
                loess_frac=loess_frac,
                n_resampling=n_resampling,
                percentile=percentile,
            )
            ax, ax_hist = plot_one_calib(
                loess=loess,
                bins=bins,
                quants=quants,
                label=label,
                ax=ax,
                color=color,
                show_bars=show_bars and (i == len(ys) - 1),
                show_hist=show_hist and (i == len(ys) - 1),
                show_tick_labels=show_tick_labels,
                hist_log_scale=hist_log_scale,
                plot_diagonal=(i == 0),
                verbose=verbose,
            )

    # Tick settings
    if not show_tick_labels:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Spine and label settings
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    # Configure legend
    legend = ax.legend(
        loc="best",
        title=f"O/E ({percentile}%CI)",
        handletextpad=0.5,
        borderaxespad=0.3,
        labelspacing=0.2,
        fancybox=False,
        frameon=True,
        edgecolor="none",
    )
    legend.get_title().set_horizontalalignment("right")
    legend.get_frame().set_facecolor(CUSTOM_PALETTES["bg_gray"][0])
    legend.get_frame().set_alpha(0.4)

    return ax, ax_hist
