import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .....general_params import watcher_config as config


def compare_code_prevalence(
    syn_data: dict,
    real_data: dict,
    n_shown: int = -1,
    ci: float = 0.95,
    size: float | int = 3,
    total_code_perv: bool = False,
):
    """Render dimension-wise plots for code prevalence in real and synthetic data
    This is designed to use data from 'count_codes_in_eval_table()' and 'bootstrap_count_codes_in_eval_tables()'.
    Args:
        syn_data (dict): Dictionary. This is supposed to be the result returned by count_codes_in_eval_table().
        real_data (dict): Dictionary. This is supposed to be the result returned by bootstrap_count_codes_in_eval_tables().
        n_shown (int): Number of data points shown per ax. If -1, everything is shown.
        ci (float): From 0.0 to 1.0. The area of the specified confidence interval is visualized in the figure.
        size (float|int): Size of each subfigrue.
        total_code_perv (bool): If true, code counts per database are devided by the number of all codes seen
            in the database on computing prevalence. Otheriwse, code counts are devided by the number of
            codes of the same group (medication, diagnosis etc).
    Returns:
        fig, axes: Matplotlib.pyplot figure and axes.
        plot_df_list: List of pandas dataframes used for plotting.
    """
    # Create axes
    fig, axes = plt.subplots(2, 3, figsize=(3 * size, 2 * size))
    fig.suptitle("Medical code prevalence")
    fig.supxlabel("Prevalence in real data [%]", y=0, va="top")
    fig.supylabel("Prevalence in synthetic data [%]", x=0.025 * 3, ha="right")
    # Plot
    lower_qtl = (1 - ci) / 2
    upper_qtl = 1 - lower_qtl
    n_patients_real = real_data["per_patient"]["total"]
    n_patients_syn = syn_data["per_patient"]["total"]
    n_total_codes_real = np.array(real_data["per_db"]["total"])
    n_total_codes_syn = syn_data["per_db"]["total"]
    plot_df_list = []
    row_no = 0
    for counting in ["per_db", "per_patient"]:
        col_no = 0
        for code_type in [config.DX_CODE, config.MED_CODE, config.LAB_CODE]:
            # Prepare dataframe
            real_df = pd.DataFrame(
                real_data[counting][code_type]["value_counts"]
            ).fillna(0)
            syn_df = pd.DataFrame(syn_data[counting][code_type]["value_counts"])
            real_df = real_df.transpose()
            syn_df = syn_df.transpose()
            # Compute prevalences from counts (counts -> prevalence)
            if counting == "per_db":
                if total_code_perv:
                    denom_db_real = n_total_codes_real
                    denom_db_syn = n_total_codes_syn
                else:
                    denom_db_real = np.array(real_data["per_db"][code_type]["n_codes"])
                    denom_db_syn = syn_data["per_db"][code_type]["n_codes"]
                real_df.loc[:, :] = real_df.values / denom_db_real
                syn_prev = syn_df.iloc[:, 0] / denom_db_syn
            else:
                real_df.loc[:, :] = real_df.values / n_patients_real
                syn_prev = syn_df.iloc[:, 0] / n_patients_syn
            # Compute mean and quantiles
            real_mean = real_df.mean(axis=1)
            real_upper = real_df.quantile(upper_qtl, axis=1)
            real_lower = real_df.quantile(lower_qtl, axis=1)
            # Organize the stats for plotting
            stats_df = pd.concat(
                [syn_prev, real_mean, real_lower, real_upper], axis=1
            ).fillna(0)
            stats_df.columns = ["syn_prev", "real_mean", "real_lower", "real_upper"]
            stats_df *= 100  # <- conversion to percents (%)
            stats_df = stats_df.sort_values("real_mean", ascending=False)
            # Plot
            if n_shown != -1:
                plot_df = stats_df.iloc[0:n_shown, :]
            else:
                plot_df = stats_df
            plot_df_list.append(plot_df)
            ax = axes[row_no, col_no]
            # NOTE: In pandas < 2.0, the following line does not work. max_lim = plot_df.max(axis=None).max() works instead.
            max_lim = plot_df.max(axis=None)
            sns.lineplot(
                x=[0, max_lim],
                y=[0, max_lim],
                color="orange",
                ax=ax,
                linestyle="dashed",
            )
            sns.scatterplot(
                x=plot_df["real_mean"], y=plot_df["syn_prev"], ax=ax, alpha=0.6, s=20
            )
            ax.fill_between(
                x=plot_df["real_mean"],
                y1=plot_df["real_lower"],
                y2=plot_df["real_upper"],
                alpha=0.2,
            )
            ax.set_aspect("equal", "box")
            ax.margins(0.05)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_title(code_type)
            # Count up ax indexes
            col_no += 1
        row_no += 1

    return fig, axes, plot_df_list
