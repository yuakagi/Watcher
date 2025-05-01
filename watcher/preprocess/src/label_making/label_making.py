"""Module to put labels to the aggregated records for training."""

from __future__ import annotations
from pandas import DataFrame
from .label_making_functions import add_labels, add_labels_single
from ....general_params import get_settings
from ....general_params import watcher_config as config
from ....utils import function_wrapper


def make_labels():
    """Assigns labels for training"""
    # Define parameters
    functions = [add_labels]
    descriptions = ["add labels"]
    section_description = "Label making"
    sheet_pattern = get_settings("PERFORMANCE_SHEET_PTN")
    json_path = sheet_pattern.replace(
        "*", "_".join(section_description.lower().split(" "))
    )

    for function, description in zip(functions, descriptions):
        function_wrapper(
            function=function,
            section_description=section_description,
            step_description=description,
            json_path=json_path,
        )

    print("All data preprocessing is completed.")


def make_labels_single(agg_df: DataFrame, model: Watcher) -> DataFrame:
    """Assigns labels for model inference"""
    # Select objects
    labels_for_codes = model.interpreter.catalogs["all_codes"][
        [config.COL_ORIGINAL_VALUE, config.COL_LABEL]
    ].copy()
    first_numeric_label = model.interpreter.min_indexes["numeric_lab_values"]
    first_timedelta_label = model.interpreter.min_indexes["timedelta"]
    n_numeric_bins = model.n_numeric_bins
    td_small_step = model.td_small_step
    td_large_step = model.td_large_step

    # Label
    labelled_df = add_labels_single(
        agg_df=agg_df,
        labels_for_codes=labels_for_codes,
        first_numeric_label=first_numeric_label,
        first_timedelta_label=first_timedelta_label,
        td_small_step=td_small_step,
        td_large_step=td_large_step,
        n_numeric_bins=n_numeric_bins,
    )
    return labelled_df
