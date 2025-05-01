"""Utility functions for model evaluation."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def roc(
    y: list | np.ndarray,
    t: list | np.ndarray,
    n_thresholds: int = 100,
    percentile: int = 95,
    n_resampling: int = 100,
):
    """Computes ROC and AUROC with resampling."""
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    sorted_index = y.argsort()
    y = y[sorted_index]
    t = t[sorted_index]
    n_items = y.shape[0]

    def _roc(y, t, n_thresholds):
        fpr, tpr = [], []
        for threshold in np.linspace(min(y), max(y), n_thresholds):
            pred = y >= threshold
            tp = (t & pred).sum()
            fn = (t & (~pred)).sum()
            tn = ((~t) & (~pred)).sum()
            fp = ((~t) & pred).sum()
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))

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

    # Compute the raw auc value
    raw_fpr, raw_tpr, raw_auc = roc(y, t, n_thresholds)

    # Compute confidence intervals by resampling
    if percentile:
        lower_p = (100 - percentile) / 2
        upper_p = 100 - lower_p
        indexes = np.arange(0, n_items)
        boot_auc_list = []
        for _ in range(n_resampling):
            # Non-parametric bootstrapping
            resampled_idx = np.random.choice(indexes, size=n_items, replace=True)
            resampled_y = y[resampled_idx]
            resampled_t = t[resampled_idx]
            sorted_new_idx = resampled_y.argsort()
            resampled_y = resampled_y[sorted_new_idx]
            resampled_t = resampled_t[sorted_new_idx]
            _, _, boot_auc = _roc(resampled_y, resampled_t, n_thresholds)
            boot_auc_list.append(boot_auc)
        # Finalize results
        boot_aucs = np.array(boot_auc_list)
        auc_low = np.percentile(boot_aucs, lower_p)
        auc_high = np.percentile(boot_aucs, upper_p)
        auc = (raw_auc, auc_low, auc_high)

    else:
        auc = (raw_auc, None, None)

    return raw_fpr, raw_tpr, auc


def calib(
    y: list | np.ndarray,
    t: list | np.ndarray,
    n_bins: int = 10,
    loess_frac: float = 0.5,
    percentile: int = 95,
    n_resampling: int = 1000,
):
    """Computes calibration curve with resampling."""

    def _compute_calib_bins(y, t, bin_lower_lims, bin_center_vals, bin_upper_lims):
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
        center_and_mean = df.groupby("center")["label"].mean()
        pred_proba = center_and_mean.index.to_numpy()
        true_freq = center_and_mean.values
        return pred_proba, true_freq

    def _quantify_calibration(y, t):
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
        # Transform y to logit scale
        y_logit = np.log(
            y / (1 - y + 1e-15)
        )  # Avoid division by zero with a small constant
        X = np.ones_like(t)  # A column of ones for the intercept
        logit_model = sm.Logit(
            t, X, offset=y_logit
        )  # Ensure `y_logit` is on the logit scale
        result = logit_model.fit(disp=False)
        calibration_in_the_large = result.params[0]
        X = sm.add_constant(y_logit)
        logit_model = sm.Logit(t, X)
        result = logit_model.fit(disp=False)
        intercept = result.params[0]
        slope = result.params[1]
        o_e_ratio = t.sum() / y.sum()
        return o_e_ratio, calibration_in_the_large, slope, intercept

    # Initialize
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    if not isinstance(t, np.ndarray):
        t = np.array(t)
    sorted_index = y.argsort()
    y = y[sorted_index]
    t = t[sorted_index]
    n_items = y.shape[0]
    indexes = np.arange(0, n_items)
    loess_xs = np.arange(0, 1, 0.005)
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
        y, t, bin_lower_lims, bin_center_vals, bin_upper_lims
    )
    raw_loess_ys = lowess(endog=t, exog=y, **loess_kw)
    o_e_ratio, calibration_in_the_large, slope, intercept = _quantify_calibration(y, t)
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
    calib_array = np.zeros((raw_calib_xs.shape[0], n_resampling))
    quant_array = np.zeros((len(quant_idx), n_resampling))
    for i in range(n_resampling):
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
        loess_ys = lowess(endog=resampled_t, exog=resampled_y, **loess_kw)
        loess_array[:, i] = loess_ys
        # Supplementary bins
        _, calib_y = _compute_calib_bins(
            resampled_y, resampled_t, bin_lower_lims, bin_center_vals, bin_upper_lims
        )
        calib_array[:, i] = calib_y
        # Quantify
        o_e_ratio, calibration_in_the_large, slope, intercept = _quantify_calibration(
            resampled_y, resampled_t
        )
        quant_array[:, i] = np.array(
            [o_e_ratio, calibration_in_the_large, slope, intercept]
        )
    # Stats
    lower_p = (100 - percentile) / 2
    upper_p = 100 - lower_p
    loess_df["low"] = np.percentile(loess_array, q=lower_p, axis=1)
    loess_df["high"] = np.percentile(loess_array, q=upper_p, axis=1)
    bins_df["low"] = np.percentile(calib_array, q=lower_p, axis=1)
    bins_df["high"] = np.percentile(calib_array, q=upper_p, axis=1)
    quant_df["low"] = np.percentile(quant_array, q=lower_p, axis=1)
    quant_df["high"] = np.percentile(quant_array, q=upper_p, axis=1)

    return loess_df, bins_df, quant_df


def calib_plot(
    y: list | np.ndarray,
    t: list | np.ndarray,
    ax,
    n_bins: int = 10,
    loess_frac: float = 0.5,
    percentile: int = 95,
    n_resampling: int = 1000,
    font="Arial",
    font_size=5,
    lw=1,
    ms=1.5,
    spine_w=0.5,
    tick_w=0.5,
    tick_s=1.5,
    tick_pad=1,
    show_tick_labels=True,
):
    """Plots calibration curve with metrics"""
    # Compute plotted values and metrics
    loess, bins, quants = calib(
        y, t, loess_frac=loess_frac, n_resampling=n_resampling, percentile=percentile
    )
    # Set general params
    plt.rcParams["font.family"] = font
    plt.rcParams["lines.linewidth"] = lw
    plt.rcParams["ytick.major.width"] = tick_w
    plt.rcParams["ytick.minor.width"] = tick_w
    plt.rcParams["xtick.major.width"] = tick_w
    plt.rcParams["xtick.minor.width"] = tick_w
    plt.rcParams["ytick.major.size"] = tick_s
    plt.rcParams["ytick.minor.size"] = tick_s
    plt.rcParams["xtick.major.size"] = tick_s
    plt.rcParams["xtick.minor.size"] = tick_s
    # Calibration curve
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")  # Ideal line for reference
    sns.lineplot(x=loess.index, y=loess["raw"], ax=ax, color="#0172b3")  # Plot LOESS
    ax.fill_between(
        loess.index,
        loess["low"],
        loess["high"],
        color="#5799d2",
        alpha=0.4,
        edgecolor=None,
    )
    y_error = np.abs(
        np.vstack([bins["low"].values, bins["high"].values]) - bins["raw"].values
    )
    ax.errorbar(
        x=bins.index.values,
        y=bins["raw"].values,
        yerr=y_error,
        capsize=ms * 1.2,
        capthick=tick_w,
        fmt="o",
        markersize=ms,
        color="#c43d3d",
    )  # Plot error bars

    oe_raw = quants.loc["oe", "raw"].item()
    oe_l = quants.loc["oe", "low"].item()
    oe_h = quants.loc["oe", "high"].item()
    citl_raw = quants.loc["citl", "raw"].item()
    citl_l = quants.loc["citl", "low"].item()
    citl_h = quants.loc["citl", "high"].item()
    slope_raw = quants.loc["slope", "raw"].item()
    slope_l = quants.loc["slope", "low"].item()
    slope_h = quants.loc["slope", "high"].item()
    intercept_raw = quants.loc["intercept", "raw"].item()
    intercept_l = quants.loc["intercept", "low"].item()
    intercept_h = quants.loc["intercept", "high"].item()
    metrics_str = (
        f"O/E: {oe_raw:.2f} ({oe_l:.2f} to {oe_h:.2f})\n"
        f"CITL: {citl_raw:.2f} ({citl_l:.2f} to {citl_h:.2f})\n"
        f"Slope: {slope_raw:.2f} ({slope_l:.2f} to {slope_h:.2f})\n"
        f"Intercept: {intercept_raw:.2f} ({intercept_l:.2f} to {intercept_h:.2f})"
    )
    # Add the textbox
    props = dict(boxstyle="round", facecolor="white", alpha=0.5, edgecolor="none")
    ax.text(
        0.05,
        1,
        metrics_str,
        transform=ax.transAxes,
        fontsize=5,
        verticalalignment="top",
        bbox=props,
    )

    # Hist
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("bottom", size=0.3, pad=0.15, sharex=ax)
    hist_data = pd.DataFrame({"y": y, "t": t})
    pos_data = hist_data.loc[hist_data["t"] == 1, "y"].values
    neg_data = hist_data.loc[hist_data["t"] == 0, "y"].values
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    pos_counts, neg_counts = [], []
    for i, lower_b in enumerate(bin_boundaries[:-1]):
        upper_b = bin_boundaries[i + 1]
        if i != n_bins:
            n_pos = ((pos_data >= lower_b) & (pos_data < upper_b)).sum()
            n_neg = ((neg_data >= lower_b) & (neg_data < upper_b)).sum()
        else:
            n_pos = ((pos_data >= lower_b) & (pos_data <= upper_b)).sum()
            n_neg = ((neg_data >= lower_b) & (neg_data <= upper_b)).sum()
        pos_counts.append(n_pos)
        neg_counts.append(n_neg)
    pos_counts = np.log10(np.maximum(pos_counts, 1))
    neg_counts = -np.log10(np.maximum(neg_counts, 1))
    bin_centers = bins.index.values
    bin_width = bin_centers[1] - bin_centers[0]
    ax_hist.axhline(0, color="black", linewidth=spine_w)
    ax_hist.bar(
        bin_centers, pos_counts, color="#de6866", edgecolor="w", width=bin_width
    )
    ax_hist.bar(
        bin_centers, neg_counts, color="#5799d2", edgecolor="w", width=bin_width
    )

    # Tick settings
    ax.tick_params(axis="x", labelsize=font_size, pad=tick_pad)
    ax.tick_params(axis="y", labelsize=font_size, pad=tick_pad)
    max_log = max(abs(neg_counts.min()), abs(pos_counts.max()))
    tick_values = np.arange(-np.ceil(max_log), np.ceil(max_log) + 1, 1)
    tick_labels = []
    for i, t_val in enumerate(tick_values):
        if (i == len(tick_values) - 1) or (i == 0):
            tick_labels.append(f"$10^{{{int(abs(t_val))}}}$")
        elif t_val == 0:
            tick_labels.append("0")
        else:
            tick_labels.append("")
    ax_hist.set_yticks(tick_values)
    ax_hist.set_yticklabels(tick_labels)
    ax_hist.tick_params(axis="x", labelsize=font_size, pad=tick_pad)
    ax_hist.tick_params(axis="y", labelsize=font_size, pad=tick_pad)

    if not show_tick_labels:
        ax_hist.set_yticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Spine and label settings
    ax.set_aspect(1)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax_hist.set_xlabel(None)
    ax_hist.set_ylabel(None)

    def set_spine_width(ax, width):
        for spine in ax.spines.values():
            spine.set_linewidth(width)

    set_spine_width(ax, spine_w)
    set_spine_width(ax_hist, spine_w)
    ax_hist.spines[["right", "top", "bottom"]].set_visible(False)
    ax_hist.get_xaxis().set_visible(False)
    ax.spines[["right", "top"]].set_visible(False)
