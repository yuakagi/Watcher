"""Module to finalize aggregated records"""

import os
import glob
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import DataFrame
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    add_distribution_columns,
    preprocess_timedelta_series,
    load_categorical_dim,
    load_special_token_index,
    flag_admission_status,
    format_df,
    tally_stats,
)


def _add_row_no(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns number of rows in the ascending order within records of a patient.

    Args:
        df (pd.DataFrame): Dataframe to be processed
    Returns:
        df (pd.DataFrame): Processed dataframe, with a new column of 'config.COL_ROW_NO'.
    """
    # Assign row numbers
    df[config.COL_ROW_NO] = range(len(df))
    return df


def _sort_lexicographically(x: pd.DataFrame) -> pd.DataFrame:
    """Sorts codes"""
    x["codes_for_sort"] = x[config.COL_ORIGINAL_VALUE].copy()
    # Make sure timedelta comes first
    # NOTE: empty string '' comes first by sort_values(ascending=True)
    is_timedelta = x[config.COL_PRIORITY] == 0
    x["codes_for_sort"] = x["codes_for_sort"].mask(is_timedelta, "")
    # Make sure lab results etc follow categoricals
    is_paired_result = x[config.COL_PRIORITY] == 2
    x["codes_for_sort"] = x["codes_for_sort"].mask(
        is_paired_result, x["codes_for_sort"].shift(1)
    )
    x = x.sort_values(
        [
            config.COL_TIMEDELTA,
            config.COL_TYPE,
            "codes_for_sort",
            config.COL_PRIORITY,
        ],
        ascending=True,
    )
    x = x.drop("codes_for_sort", axis=1)
    return x


def _finalization_core(
    df: DataFrame,
    categorical_dim: int,
    eot_idx: int = None,
    single_patient: bool = False,
) -> tuple[DataFrame, dict]:
    """Core finalization steps"""
    # Initialize
    stats = {}
    dmg_type_no = config.RECORD_TYPE_NUMBERS[config.DMG]
    categorical_cols = [f"c{i}" for i in range(categorical_dim)]
    if not single_patient:
        if eot_idx is None:
            raise ValueError(
                "Argument 'eot_idx' must not be none during dataset preparation."
            )

    # **************************************************
    # * Exclude rows outside the training and test span*
    # **************************************************
    if not single_patient:
        # Add columns for temporal data distribution
        original_cols = df.columns.tolist()
        df = add_distribution_columns(df)
        # Masking by periods
        train_period_mask = df[config.COL_TRAIN_PERIOD] == 1
        test_period_mask = df[config.COL_TEST_PERIOD] == 1
        # Keep demographic data
        dmg_mask = df[config.COL_TYPE] == dmg_type_no
        inclusion_mask = train_period_mask | test_period_mask | dmg_mask
        # Drop the added columns
        df = df[original_cols]
        # Count the excluded records
        invalid_record_stats = {}
        n_total_invalid_records = 0
        invalid_records = ~inclusion_mask
        for record_type, type_no in config.RECORD_TYPE_NUMBERS.items():
            type_mask_inv = df[config.COL_TYPE] == type_no
            n_invalid_records = df.loc[
                invalid_records & type_mask_inv, config.COL_RECORD_ID
            ].nunique()
            invalid_record_stats[record_type] = n_invalid_records
            n_total_invalid_records += n_invalid_records
        invalid_record_stats["total"] = n_total_invalid_records
        stats["n records outside both train and test periods"] = (
            invalid_record_stats.copy()
        )
        # Drop records
        df = df.loc[inclusion_mask, :]

    # ***********************************************
    # * Extract patients only with demographic rows *
    # ***********************************************
    if not single_patient:
        original_n_patients = df[config.COL_PID].nunique()
        id_counts = df.value_counts(config.COL_PID)
        id_selction_df = pd.DataFrame({"count": id_counts})
        id_selction_df = id_selction_df[
            id_selction_df["count"] > config.DEMOGRAPHIC_ROWS
        ]
        df = pd.merge(df, id_selction_df, on=config.COL_PID, how="inner")
        # Count the number of excluded patients
        n_invalid_patients = original_n_patients - df[config.COL_PID].nunique()
        stats["n patients only with demographics"] = n_invalid_patients

    # *******************************
    # * Handle the demographic rows *
    # *******************************
    # Description:
    #   This process handles timestamp and timedelta values of demographic rows to ensure that
    #   timelines always start with 00:00.

    # Detect demgraphic rows
    dmg_mask = df[config.COL_TYPE] == dmg_type_no
    dmg_timedelta_mask = dmg_mask & (~df[config.COL_YEARS].isna())
    # Fill timestamp and timedelta related to demgraphic values with None for replacement
    df.loc[dmg_mask, [config.COL_TIMEDELTA, config.COL_TIMESTAMP]] = None
    # Fill timestamp and timedelta values with the oldest values in each patient timeline
    df[[config.COL_TIMEDELTA, config.COL_TIMESTAMP]] = df[
        [config.COL_TIMEDELTA, config.COL_TIMESTAMP]
    ].bfill()
    # Round hh:mm compartments of timestamp and timedelta values to 00:00
    for t_col in [config.COL_TIMESTAMP, config.COL_TIMEDELTA]:
        df.loc[dmg_mask, t_col] = df.loc[dmg_mask, t_col].dt.floor(freq="D")
    # Preprocess timedelta values for model input
    df.loc[dmg_timedelta_mask, config.TIMEDELTA_COMPONENT_COLS] = (
        preprocess_timedelta_series(
            df.loc[dmg_timedelta_mask, config.COL_TIMEDELTA]
        ).values
    )

    # ********************************
    # * Sort codes lexicographically *
    # ********************************
    df = df.groupby(config.COL_PID, group_keys=False).apply(_sort_lexicographically)

    # **********************
    # * Assign row numbers *
    # **********************
    df = df.groupby(config.COL_PID).apply(_add_row_no)

    # ***********************
    # * Add the final rows  *
    # * (with [EOT] tokens) *
    # ***********************
    # NOTE: For single patient processing, rows for [EOT] are not appended.
    if not single_patient:
        # Ensure sorting
        df = df.reset_index(drop=True).sort_values([config.COL_PID, config.COL_ROW_NO])
        # Initialize the appended set of rows as a dataframe
        appended_rows_df = df.drop_duplicates(subset=config.COL_PID, keep="last").copy()
        # Add the embedding index of [EOT] token
        appended_rows_df[categorical_cols[0]] = eot_idx
        # Pad input values
        appended_rows_df[categorical_cols[1:]] = 0
        appended_rows_df[config.COL_NUMERIC] = np.nan
        appended_rows_df[config.TIMEDELTA_COMPONENT_COLS] = np.nan
        # Fill config.COL_TYPE col with the type number of EOT
        eot_type_no = config.RECORD_TYPE_NUMBERS[config.EOT]
        appended_rows_df[config.COL_TYPE] = eot_type_no
        # Fill the original value col with [EOT]
        appended_rows_df[config.COL_ORIGINAL_VALUE] = "[EOT]"
        # Fill record IDs
        appended_rows_df[config.COL_RECORD_ID] = (
            appended_rows_df[config.COL_PID] + "EOT"
        )
        # Add +1 to the row numbers
        appended_rows_df[config.COL_ROW_NO] += 1
        # Append rows
        appended_rows_df = appended_rows_df.astype(df.dtypes.to_dict())
        df = pd.concat([df, appended_rows_df])
        # Sort values so that the appended rows are alinged at ends of timelines
        df = df.reset_index(drop=True)
        df = df.sort_values([config.COL_PID, config.COL_ROW_NO])

    # *************************************
    # * Add a column for admission status *
    # *************************************
    df[config.COL_ADM] = flag_admission_status(df, ignore_truncated=False)

    # *********************************************
    # * Final datatype check and column selection *
    # *********************************************
    df = format_df(df=df, table_params=config.FINAL_AGG_TABLE_COLS)

    return df, stats


def _finalize_aggregated_files(file: str, categorical_dim: int, eot_idx: int) -> dict:
    """Finalizes the aggregated records.

    The main operations this function performs:
        1. Exclude records outside the study period
        2. Exclude patients only with demographic records
        3. Cleans timestamp and timedelta values for demographic records
        4. Assign row numbers
        5. Append [EOT] tokens
    """
    # Load table
    df = pd.read_pickle(file)

    # Apply finalization core steps
    df, stats = _finalization_core(
        df,
        categorical_dim=categorical_dim,
        eot_idx=eot_idx,
        single_patient=False,
    )

    # Save
    basename = os.path.basename(file).replace("aggregated", "finalized_aggregated")
    output_file_path = os.path.join(get_settings("TEMP_DIR"), basename)
    df.to_pickle(output_file_path)

    return stats


def finalize_aggregated_files():
    """Finalizes the aggregated records."""
    categorical_dim = load_categorical_dim()
    eot_idx = load_special_token_index("[EOT]")

    # Collect aggregated files
    aggregated_files = [
        os.path.join(get_settings("AGGREGATED_TABLES_DIR"), file_name)
        for file_name in os.listdir(get_settings("AGGREGATED_TABLES_DIR"))
    ]

    # Execute parallelism
    max_workers = get_settings("MAX_WORKERS")
    stats_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _finalize_aggregated_files,
                file=file,
                categorical_dim=categorical_dim,
                eot_idx=eot_idx,
            )
            for file in aggregated_files
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing tasks"
        ):
            stats = future.result()
            stats_list.append(stats)
    final_stats = tally_stats(stats_list)

    # Clear existing files
    fin_aggregated_dir = get_settings("FIN_AGGREGATED_TABLES_DIR")
    for file in os.listdir(fin_aggregated_dir):
        os.remove(os.path.join(fin_aggregated_dir, file))

    # Move product files from the temp dir to the target dir
    for src in glob.glob(os.path.join(get_settings("TEMP_DIR"), "*.pkl")):
        dst = os.path.join(fin_aggregated_dir, os.path.basename(src))
        shutil.move(src, dst)

    return final_stats


def finalize_aggregated_single(agg_df: DataFrame, categorical_dim: int) -> DataFrame:
    """Finalizes aggregated records from a patient for model inference."""
    agg_df, _ = _finalization_core(
        agg_df,
        categorical_dim=categorical_dim,
        single_patient=True,
        eot_idx=None,
    )
    return agg_df
