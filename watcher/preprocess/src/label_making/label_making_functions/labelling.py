"""Functions to put labels"""

import os
import json
import glob
from typing import Literal
import shutil
import numpy as np
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    compute_numeric_steps,
    create_labels_from_timedelta_pairs,
    add_distribution_columns,
    parallel_map_partitions,
    load_catalog,
    tally_stats,
    format_df,
)


def label_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns labels on special tokens.
    Args:
        df (pd.DataFrame): Dataframe to be processed
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    # Extract token rows (Single tokens have non-zero values in 'c1' column, and all zeros in other categorical columns
    df[["c0", "c1"]] = df[["c0", "c1"]].astype(int)
    c0_nonzero_mask = df["c0"] != 0
    c1_zero_mask = df["c1"] == 0
    token_rows = c0_nonzero_mask & c1_zero_mask

    # Copy values to the config.COL_LABEL col
    df[config.COL_LABEL] = df[config.COL_LABEL].astype(int)
    df.loc[token_rows, config.COL_LABEL] = df.loc[token_rows, "c0"].copy()

    return df


def label_codes(
    df: pd.DataFrame, labels_for_codes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Assigns labels on code values (such as diagnosis and medication).
    Args:
        df (pd.DataFrame): Dataframe to be processed.
        labels_for_codes (pd.DataFrame): Catalog that contains codes and labels.
            This catalog can be loaded by 'load_catalog' function.
            This catalog can be explicitly passed as an argument, which allows reuse of this table.
            Default is None, which forces this function to load the table.
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    # Load the label catalog if not passed as the argument
    if labels_for_codes is None:
        code_catalog = load_catalog(
            catalog_type="all_codes", catalogs_dir=get_settings("CATALOGS_DIR")
        )
        labels_for_codes = code_catalog[
            [config.COL_ORIGINAL_VALUE, config.COL_LABEL]
        ].copy()

    # Data type conversion
    df[config.COL_LABEL] = df[config.COL_LABEL].astype(int)
    labels_for_codes[config.COL_LABEL] = labels_for_codes[config.COL_LABEL].astype(int)

    # Rename column for mapping
    labels_for_codes = labels_for_codes.rename(columns={config.COL_LABEL: "code_label"})

    # Assign labels
    df = pd.merge(
        left=df,
        right=labels_for_codes[[config.COL_ORIGINAL_VALUE, "code_label"]],
        on=config.COL_ORIGINAL_VALUE,
        how="left",
    )

    # Copy values to the config.COL_LABEL col
    code_mask = ~(df["code_label"].isna())
    df.loc[code_mask, config.COL_LABEL] = df.loc[code_mask, "code_label"].astype(int)
    df = df.drop("code_label", axis=1)

    return df


def label_numerics(
    df: pd.DataFrame,
    n_numeric_bins: int,
    first_numeric_label: int,
) -> pd.DataFrame:
    """Assigns labels on numeric values.
    Args:
        df (pd.DataFrame): Dataframe to be processed
        n_numeric_bins (int): Number of bins for numeric values.
        first_numeric_label (int): First index of numericl labels
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    num_mask = df[config.COL_NUMERIC].notna()

    # <--- Labeling using percentiles --->
    numeric_steps = compute_numeric_steps(num_bins=n_numeric_bins)
    numeric_steps = numeric_steps.reshape(-1, 1)
    # Label data in batches
    # NOTE: This step is memory-intensive.
    patient_batch_size = config.PATIENT_BATCH_SIZE_FOR_LAB
    unique_patients = df[config.COL_PID].unique()
    for i in range(0, len(unique_patients), patient_batch_size):
        batch_patients = unique_patients[i : i + patient_batch_size]
        batch_mask = df[config.COL_PID].isin(batch_patients) & num_mask
        if batch_mask.any():
            num_series = df.loc[batch_mask, config.COL_NUMERIC].values
            num_indexes = np.argmin(np.abs(numeric_steps - num_series), axis=0)
            numeric_labels = num_indexes + first_numeric_label
            numeric_labels = numeric_labels.astype(int)
            # Assign
            df.loc[batch_mask, config.COL_LABEL] = numeric_labels

    return df


def label_timedelta(
    df: pd.DataFrame, first_timedelta_label: int, small_step: int, large_step: int
) -> pd.DataFrame:
    """Assigns labels on timedelta values.
    Args:
        df (pd.DataFrame): Dataframe to be processed
        first_timedelta_label (int): First index of timdelta labels
        small_step (int): Small step for timedelta labelling
        large_step (int): Large step for timedelta labelling
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    # Mask
    age_row_mask = df[config.COL_YEARS].notna()
    first_row_mask = (
        df[config.COL_ROW_NO] == 0
    )  # Ignore the first rows (start of a trajectory)
    timedelta_changing_point = age_row_mask & (~first_row_mask)

    # Create a pair of timedelta cols
    df[config.COL_TIMEDELTA] = pd.to_timedelta(df[config.COL_TIMEDELTA])
    shifted_timedelta_col = df[config.COL_TIMEDELTA].shift(1)

    # Mask irrelevant timedeltas
    masked_timedeltas = df[config.COL_TIMEDELTA].mask(~timedelta_changing_point, pd.NaT)
    masked_shifted = shifted_timedelta_col.mask(~timedelta_changing_point, pd.NaT)

    # Create labels using the two timedelta series
    final_labels = create_labels_from_timedelta_pairs(
        masked_timedeltas, masked_shifted, first_timedelta_label, small_step, large_step
    )

    # Add timedelta labels to the dataframe
    df.loc[timedelta_changing_point, config.COL_LABEL] = final_labels[
        timedelta_changing_point
    ]

    return df


def _add_labels_core(
    df: pd.DataFrame,
    labels_for_codes: pd.DataFrame,
    first_numeric_label: int,
    first_timedelta_label: int,
    td_small_step: int,
    td_large_step: int,
    n_numeric_bins: int,
    flag: Literal["train", "validation", "test", "single"],
) -> pd.DataFrame:
    """Core steps of labelling."""

    # Initialize columns
    df[config.COL_LABEL] = config.LOGITS_IGNORE_INDEX

    # Label code values (such as diagnosis)
    df = label_codes(df, labels_for_codes)
    # Label special tokens (including nonnumeric laboratory test values)
    df = label_tokens(df)
    # Label numeric values
    df = label_numerics(
        df, first_numeric_label=first_numeric_label, n_numeric_bins=n_numeric_bins
    )
    # Label timedelta values
    df = label_timedelta(
        df,
        first_timedelta_label=first_timedelta_label,
        small_step=td_small_step,
        large_step=td_large_step,
    )

    # Set the ignored label index to the beginning of a timeline
    starting_points = df[config.COL_ROW_NO] == 0
    df.loc[starting_points, config.COL_LABEL] = config.LOGITS_IGNORE_INDEX

    # Prevent learning [EOT] during admissions
    if flag != "single":
        df.loc[
            (df[config.COL_ORIGINAL_VALUE] == "[EOT]") & (df[config.COL_ADM] == 1),
            config.COL_LABEL,
        ] = config.LOGITS_IGNORE_INDEX

    # Check effective token lengths of train patients, and validate them for inclusion
    if flag != "single":
        # Determine col
        if flag in ["train", "validation"]:
            period_col = config.COL_TRAIN_PERIOD
        else:
            period_col = config.COL_TEST_PERIOD
        # Check effective length
        df = add_distribution_columns(df)
        dmg_mask = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DMG]
        period_mask = df[period_col].astype(int).astype(bool)
        df["effective"] = period_mask | dmg_mask
        df["effective_length"] = df.groupby(config.COL_PID)["effective"].transform(
            "sum"
        )
        df[config.COL_INCLUDED] = (
            df["effective_length"] >= get_settings("MIN_TRAJECTORY_LENGTH")
        ).astype(int)
        df = df.drop(["effective", "effective_length"], axis=1)

    # *********************************************
    # * Final datatype check and column selection *
    # *********************************************
    df = format_df(df=df, table_params=config.LABELLED_TABLE_COLS)

    return df


def _add_labels(
    df: pd.DataFrame,
    labels_for_codes: pd.DataFrame,
    first_numeric_label: int,
    first_timedelta_label: int,
    td_small_step: int,
    td_large_step: int,
    n_numeric_bins: int,
    file_name: str,
    temp_path_pattern: str,
    flag: Literal["train", "validation", "test", "single"],
) -> tuple[pd.DataFrame, list[str]]:
    """Helper function to add labels."""

    df = _add_labels_core(
        df=df,
        labels_for_codes=labels_for_codes,
        first_numeric_label=first_numeric_label,
        first_timedelta_label=first_timedelta_label,
        td_small_step=td_small_step,
        td_large_step=td_large_step,
        n_numeric_bins=n_numeric_bins,
        flag=flag,
    )

    # Save the file
    # NOTE: File is named in a way that its name matches the source aggregated table.
    file_name_parts = file_name.split("_")
    file_no = file_name_parts[-1].replace(".pkl", "")
    temp_path = temp_path_pattern.replace("*", file_no)
    df.to_pickle(temp_path)

    # Collect included patient IDs
    included_ids = {
        "ids": df.loc[df[config.COL_INCLUDED] == 1, config.COL_PID].unique().tolist()
    }

    return df, included_ids


def add_labels():
    """Labels for training to the aggregated tables.

    Files are saved in get_settings(LABELLED_TABLES_DIR).
    Files are named 'labelled_<train/validation/test>_<file number>.pkl'.
    """
    # Load labelling catalog for code values
    code_catalog = load_catalog(
        catalog_type="all_codes", catalogs_dir=get_settings("CATALOGS_DIR")
    )
    labels_for_codes = code_catalog[
        [config.COL_ORIGINAL_VALUE, config.COL_LABEL]
    ].copy()
    # Load first label indexes
    numeric_catalog = load_catalog(
        catalog_type="numeric_lab_values", catalogs_dir=get_settings("CATALOGS_DIR")
    )
    timedelta_catalog = load_catalog(
        catalog_type="timedelta", catalogs_dir=get_settings("CATALOGS_DIR")
    )
    td_small_step = get_settings("TD_SMALL_STEP")
    td_large_step = get_settings("TD_LARGE_STEP")
    n_numeric_bins = get_settings("NUMERIC_BINS")
    first_numeric_label = numeric_catalog[config.COL_LABEL].iloc[0]
    first_timedelta_label = timedelta_catalog[config.COL_LABEL].iloc[0]
    # Ensure that the temporary file directory is empty
    for old_file in os.listdir(get_settings("TEMP_DIR")):
        os.remove(os.path.join(get_settings("TEMP_DIR"), old_file))
    # Execute the main steps
    output_dir = get_settings("LABELLED_TABLES_DIR")
    id_dir = get_settings("PID_DIR")

    included_patient_id_dict = {}
    for flag in [config.TRAIN, config.VAL, config.TEST]:
        file_pattern = get_settings("FIN_AGGREGATED_FILE_PTN")
        source_path_pattern = file_pattern.replace("*", f"{flag}_*")
        temp_path_pattern = os.path.join(
            get_settings("TEMP_DIR"),
            config.LABELLED_FILE_PATTERN.replace("*", f"{flag}_*"),
        )

        stats = parallel_map_partitions(
            source_path_pattern=source_path_pattern,
            clear_temp_dir=False,  # <- Ensure not to delete temp files
            function=_add_labels,
            chunksize=-1,  # <- Do not change the chunksize in this function
            file_name_as_arg=True,
            labels_for_codes=labels_for_codes,
            first_numeric_label=first_numeric_label,
            first_timedelta_label=first_timedelta_label,
            td_small_step=td_small_step,
            td_large_step=td_large_step,
            n_numeric_bins=n_numeric_bins,
            temp_path_pattern=temp_path_pattern,
            flag=flag,
        )
        included_patient_ids = tally_stats(stats)
        included_patient_id_dict[flag] = included_patient_ids["ids"]

    # Save included patient IDs
    with open(
        os.path.join(id_dir, config.INCLUDED_PATIENT_ID_LIST), "w", encoding="utf-8"
    ) as f:
        json.dump(included_patient_id_dict, f, indent=2)

    # Move files
    for old_file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, old_file))
    temp_path_pattern = os.path.join(
        get_settings("TEMP_DIR"), config.LABELLED_FILE_PATTERN
    )
    for temp_file in glob.glob(temp_path_pattern):
        dst = os.path.join(output_dir, os.path.basename(temp_file))
        shutil.move(temp_file, dst)


def add_labels_single(
    agg_df: pd.DataFrame,
    labels_for_codes: pd.DataFrame,
    first_numeric_label: int,
    first_timedelta_label: int,
    td_small_step: int,
    td_large_step: int,
    n_numeric_bins: int,
):
    """Assigns labels for model inference."""
    labelled_df = _add_labels_core(
        df=agg_df,
        labels_for_codes=labels_for_codes,
        first_numeric_label=first_numeric_label,
        first_timedelta_label=first_timedelta_label,
        n_numeric_bins=n_numeric_bins,
        td_small_step=td_small_step,
        td_large_step=td_large_step,
        flag="single",
    )
    return labelled_df
