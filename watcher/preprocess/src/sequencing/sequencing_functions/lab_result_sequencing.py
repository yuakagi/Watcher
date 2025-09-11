"""Module to align laboratory test result records"""

import os
import numpy as np
import pandas as pd
from .sequencing_core import sequence_core
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    render_token_map,
    convert_codes_to_indexes,
    parallel_map_partitions,
    load_categorical_dim,
    load_special_token_dict,
    load_lab_percentiles,
    load_nonnumeric_stats,
    load_numeric_stats,
    compute_numeric_steps,
    tally_stats,
)


def _sequence_lab_result_records(
    df: pd.DataFrame,
    task_no: int,
    categorical_dim: int,
    n_numeric_bins: int,
    percentiles: pd.DataFrame,
    percentile_cols: list,
    nonnum_stats: pd.DataFrame,
    num_stats: pd.DataFrame,
    lab_token_map: pd.DataFrame,
    special_token_dict: dict,
) -> pd.DataFrame:
    """Helper function that performs preprocessing steps for laboratory test result records.

    Args:
        df (pd.DataFrame): Dataframe to be processed
        task_no (int): number of current task processed by the child process.
            This is passed by 'parallel_map_partition' function.
        categorical_dim (int): Maximum number of embedding indexes to represent a code value
        percentiles (pd.DataFrame): Table containing mean values and standard deviation.
            This table is used to calculate z-scores.
        nonnum_stats (pd.DataFrame): Table containing unique nonnumeric laboratory test values and corresponding tokens by
            lab test codes. This table is necessary for giving embedding indexes for nonnumeric test result values.
        num_stats (pd.DataFrame): Stats on numeric values, including means and stds.
        lab_token_map (pd.DataFrame): Table to map lab test codes into series of embedding indexes
        special_token_dict (dict): Dictionary to convert nonnumeric test result tokens into embedding indexes
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    stats = {}
    # Merge tables (numeric stats, nonnumeric stats)
    df = df.merge(
        percentiles[
            [config.COL_ITEM_CODE, "unit", "count"]
        ],  # <- counts only, at this point
        on=[config.COL_ITEM_CODE, "unit"],
        how="left",
    )
    df = df.rename(columns={"count": "num_count"})
    df = df.merge(
        nonnum_stats, on=[config.COL_ITEM_CODE, config.COL_NONNUMERIC], how="left"
    )
    df = df.rename(columns={"count": "nonnum_count"})
    # Drop duplicated lab test codes within the same laboratory test result report
    df["count"] = df["num_count"].fillna(df["nonnum_count"])
    df = df.sort_values(
        [config.COL_PID, config.COL_TIMEDELTA, config.COL_ITEM_CODE, "count"],
        ascending=[True, True, True, False],
    ).drop_duplicates(
        subset=[config.COL_PID, config.COL_TIMEDELTA, config.COL_ITEM_CODE],
        keep="first",
    )
    df = df.drop(["count", "num_count", "nonnum_count"], axis=1)

    # ********** Categorical values to vocabulary indexes ***************
    # You need to create the 'categorical_cols', and COL_ORIGINAL_VALUE here.
    # NOTE: Replace oov values with the oov index, not tull
    df = convert_codes_to_indexes(df, lab_token_map, config.COL_ITEM_CODE)
    df[config.COL_ORIGINAL_VALUE] = df[config.COL_ITEM_CODE].copy()
    # *******************************************************************

    # ********** Process numeric values *********************************
    # You need to create the COL_NUMERIC, and COL_ORIGINAL_NUMERIC here.
    # Copy original lab result values first
    df[config.COL_ORIGINAL_NUMERIC] = (
        df[config.COL_NUMERIC].astype(str) + " " + df["unit"].fillna("")
    )

    # Select rows
    num_mask = df[config.COL_NUMERIC].notna()

    # <--- Normalization using percentiles --->
    # Compute normalized vals
    numeric_steps = compute_numeric_steps(num_bins=n_numeric_bins)
    # NOTE: This section is slow, consider other implementation.
    patient_batch_size = config.PATIENT_BATCH_SIZE_FOR_LAB
    unique_patients = df[config.COL_PID].unique()
    for i in range(0, len(unique_patients), patient_batch_size):
        batch_patients = unique_patients[i : i + patient_batch_size]
        # Create a mask for selected patients
        p_num_mask = df[config.COL_PID].isin(batch_patients) & num_mask
        if p_num_mask.any():
            # Create a new dataframe for mapping percentiles
            num_df = df.loc[
                p_num_mask, [config.COL_ITEM_CODE, "unit", config.COL_NUMERIC]
            ].copy()
            num_df = num_df.merge(
                percentiles, on=[config.COL_ITEM_CODE, "unit"], how="left"
            )
            # Check for the mapping
            percentiled = num_df[percentile_cols].notna().all(axis=1)
            if percentiled.any():
                # Transform values to percentiles
                vals = num_df.loc[percentiled, config.COL_NUMERIC].values.reshape(-1, 1)
                pvs = num_df.loc[percentiled, percentile_cols].values  # 2D array
                diff = np.abs(pvs - vals)
                # NOTE: When duplicated percentile candidates exist, the smallest index is chosen here by 'np.argmin()'.
                converted_vals = numeric_steps[np.argmin(diff, axis=1)]
                # Assign the values back to the dataframe
                num_df["normalized_vals"] = np.nan
                num_df.loc[percentiled, "normalized_vals"] = converted_vals
                # Replace the numerics with percentiles in the original dataframe
                df.loc[p_num_mask, config.COL_NUMERIC] = num_df[
                    "normalized_vals"
                ].values
            else:
                df.loc[p_num_mask, config.COL_NUMERIC] = np.nan

    # Finally, unprocessed items are all replaced with nans
    df.loc[~num_mask, config.COL_NUMERIC] = np.nan
    # *******************************************************************

    # ********** Nonnumeric values to vocabulary indexes ****************
    # You need to create the COL_NONNUMERIC, and COL_ORIGINAL_NONNUMERIC here.
    # NOTE: Replace oov values with the oov index, not null.
    # Copy original lab result values first
    df[config.COL_ORIGINAL_NONNUMERIC] = df[config.COL_NONNUMERIC].copy()
    special_token_df = pd.DataFrame.from_dict(
        special_token_dict, orient="index", columns=["nonnum_index"]
    )
    df = df.merge(
        special_token_df, left_on=config.COL_TOKEN, right_index=True, how="left"
    )
    df = df.drop(config.COL_NONNUMERIC, axis=1)
    # NOTE: COL_NONNUMERIC's dtype is string up to this point, then integer after this point
    df = df.rename(columns={"nonnum_index": config.COL_NONNUMERIC})
    oov_nonnum_mask = ~num_mask & df[config.COL_NONNUMERIC].isna()
    df.loc[oov_nonnum_mask, config.COL_NONNUMERIC] = config.OOV_INDEX
    # *******************************************************************

    # Drop unnecessary cols
    df = df.drop(config.COL_ITEM_CODE, axis=1)

    # Main
    df, substats = sequence_core(
        df=df,
        task_no=task_no,
        categorical_dim=categorical_dim,
        type_no=config.RECORD_TYPE_NUMBERS[config.LAB_R],
        dropna_subset=[
            config.COL_PID,
            config.COL_TIMEDELTA,
            config.COL_ORIGINAL_VALUE,
            "c0",
        ],
    )

    stats = {**stats, **substats}

    return df, stats


def sequence_lab_result_records() -> dict:
    """Aligns records linearly before creating patient timeline matrices to laboratory test result records.
    See the helper function for details.
    """
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.LAB_RESULT_TABLE_PATTERN
    )
    output_file_path = os.path.join(
        get_settings("SEQUENCED_TABLES_DIR"),
        config.LAB_RESULT_TABLE_PATTERN.replace("*", "sequenced"),
    )
    n_numeric_bins = get_settings("NUMERIC_BINS")

    # Get the categorical_dim and special token reference
    special_token_dict = load_special_token_dict()
    categorical_dim = load_categorical_dim()

    # Load tokenization maps.
    lab_token_map = render_token_map(map_type=config.LAB_CODE)

    # Load lab numeric stats / percentiles
    percentiles, percentile_cols = load_lab_percentiles(single_unit=False)
    num_stats = load_numeric_stats(single_unit=False)

    # Load lab nonnumeric stats.
    nonnum_stats = load_nonnumeric_stats()
    nonnum_stats = nonnum_stats[
        [config.COL_ITEM_CODE, config.COL_NONNUMERIC, config.COL_TOKEN, "count"]
    ]

    stats_list = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_sequence_lab_result_records,
        output_file_path=output_file_path,
        chunksize=-1,
        single_file=False,
        task_no_as_arg=True,
        categorical_dim=categorical_dim,
        n_numeric_bins=n_numeric_bins,
        percentiles=percentiles,
        percentile_cols=percentile_cols,
        nonnum_stats=nonnum_stats,
        num_stats=num_stats,
        lab_token_map=lab_token_map,
        special_token_dict=special_token_dict,
    )
    return tally_stats(stats_list)


def sequence_lab_result_records_single(
    df: pd.DataFrame,
    categorical_dim: int,
    n_numeric_bins: int,
    percentiles: pd.DataFrame,
    percentile_cols: list,
    nonnum_stats: pd.DataFrame,
    num_stats: pd.DataFrame,
    lab_token_map: pd.DataFrame,
    special_token_dict: dict,
) -> pd.DataFrame:
    """Aligns records linearly before creating patient timeline matrices.
    This is designed to handle data for a single patient.
    See the helper function for details.
    """
    if df.size:
        df, _ = _sequence_lab_result_records(
            df=df,
            task_no=0,
            categorical_dim=categorical_dim,
            n_numeric_bins=n_numeric_bins,
            percentiles=percentiles,
            percentile_cols=percentile_cols,
            nonnum_stats=nonnum_stats,
            num_stats=num_stats,
            lab_token_map=lab_token_map,
            special_token_dict=special_token_dict,
        )
        return df

    else:
        return None
