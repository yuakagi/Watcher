"""Functions to make stats for laboratory test results."""

import os
import glob
import numpy as np
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    parallel_map_partitions,
    filter_records,
    tally_stats,
    discretize_percentiles,
)


def _tokenize_nonnumerics(lab_test_group: pd.DataFrame) -> pd.DataFrame:
    """Tokenizes nonnumeric laboratory test result values.

    This function consists of the following steps:
        - 1. Assign tokens to very common nonnumeric values (such as +ve and -ve)
            (This step uses a dictionary for tokenization,
                which is defined in 'config.POS_NEG_DICT'.)
        - 2. Give '[top n]' tokens to other distinct nonnumeric values by frequencies.
            (Take blood type test for example; if the most frequently seen result is 'A', followed by
             'O', 'B' and 'AB', then 'A' is tokenized to '[top1]', 'O' to '[top2]', 'B' to '[top3]' and
             'AB' to '[top4]'.
             This algorithm is designed to reduce the vocabulary size for nonnumeric test result values.)
    Args:
        lab_test_group (pd.DataFrame): Each dataframe contains nonnumeric values of a unique laboratory test code.
    Returns:
        lab_test_group (pd.DataFrames): Processed dataframe
    """
    # Rank the nonnumeric values by count
    lab_test_group = lab_test_group.sort_values("count", ascending=False).reset_index(
        drop=True
    )

    # Tokenize very common nonnumeric values first, such as '(+)' and '(-).
    lab_test_group[config.COL_TOKEN] = lab_test_group[config.COL_NONNUMERIC].copy()
    lab_test_group[config.COL_TOKEN] = lab_test_group[config.COL_TOKEN].replace(
        config.POS_NEG_DICT
    )

    # Create a dictionary to map nonnumeric values to tokens
    rank_dict = {}
    ranks = []
    preset_tokens = set(config.POS_NEG_DICT.values())
    for rank, row in lab_test_group.iterrows():
        # Iterate through rows in the order of the frequencies
        ranks.append(rank)
        val = row[config.COL_TOKEN]
        if (val not in rank_dict) and (val not in preset_tokens):
            # Here, a 'token' value is a copied nonnumeric value from 'config.COL_NONNUMERIC' col
            rank_dict[val] = f"[top{rank+1}]"

    # Tokenize nonnumeric values other than common ones by ranks (frequencies)
    lab_test_group[config.COL_TOKEN] = lab_test_group[config.COL_TOKEN].replace(
        rank_dict
    )

    # Ensure that 'top*' tokens and preset tokens are not mixed within a unique laboratory test code
    # For example, if a laboratory test code has nonnumeric value tokens [pos], [neg], [top1] and [top2],
    # then these are replaced to [top1], [top2], [top3] and [top4]
    all_tops = np.array([f"[top{rank+1}]" for rank in ranks])
    unique_tokens = np.array(lab_test_group[config.COL_TOKEN].tolist())
    flag = (all_tops == unique_tokens).sum()
    if not (flag == 0 or flag == len(unique_tokens)):
        replacement_dict = dict(zip(unique_tokens, all_tops))
        lab_test_group[config.COL_TOKEN] = lab_test_group[config.COL_TOKEN].replace(
            replacement_dict
        )

    # Add 'unique count'.
    lab_test_group["unique_count"] = lab_test_group[config.COL_TOKEN].nunique()

    return lab_test_group


def _tally_nonnums(
    df: pd.DataFrame,
    patient_id_dict: dict,
    pid: int,
) -> None:
    """Performs the following steps:
        - 1. Filter records
        - 2. Tally unique nonnumeric test results by laboratory test codes.
    Args:
        df (pd.DataFrame): Target dataframe
        patient_id_dict (dict): List of dictionaries with each subdictionary contain patient IDs.
            Patient ID dictionaries are stored in the order of train, validation and test. This list can be
            directly passed by 'parallel_map_partition' with 'patient_id_dict_as_arg = True'.
        pid (int): Child process ID. This is passed from 'parallel_map_partitions'.
    Returns:
        None
        (Intermediate tables are saved as pickled objects so that they can be aggregated later.)
    """
    # Filter and sort
    train_ids = patient_id_dict["train"]
    df = filter_records(df, train_ids, period="train")
    df = df.sort_values(config.COL_ITEM_CODE)
    mask = df[config.COL_NUMERIC].isna()
    nonnum_df = df.loc[mask, :]

    # Tally unique nonnumeric test result values by laboratory test codes
    grouped_nonnum_stats = nonnum_df.groupby([config.COL_ITEM_CODE])[
        config.COL_NONNUMERIC
    ].value_counts()
    grouped_nonnum_stats.name = "count"
    grouped_nonnum_stats = grouped_nonnum_stats.reset_index()

    # Save the intermediate results
    nonnum_stats_temp_path = get_settings("TEMP_NONNUM_STATS_PTN").replace(
        "*", str(pid)
    )
    if os.path.exists(nonnum_stats_temp_path):
        # Load and concatenate tables if previous one exists
        old_nonnum_stats = pd.read_pickle(nonnum_stats_temp_path)
        grouped_nonnum_stats = pd.concat([old_nonnum_stats, grouped_nonnum_stats])
    grouped_nonnum_stats.to_pickle(nonnum_stats_temp_path)

    return None


def _tally_nums(df: pd.DataFrame, patient_id_dict: dict) -> tuple[pd.DataFrame, dict]:
    """Tally numeric test results grouped by item code and unit.
    A dataframe is returned for consistency.
    """
    # Filter records
    train_ids = patient_id_dict["train"]
    df = filter_records(df, train_ids, period="train")
    num_df = df.loc[df[config.COL_NUMERIC].notna(), :].copy()

    # Handle missing units
    num_df["unit"] = num_df["unit"].fillna("")

    # Group by item code and unit, aggregate values into lists
    grouped = num_df.groupby([config.COL_ITEM_CODE, "unit"])[config.COL_NUMERIC].agg(
        list
    )

    # Convert to proper nested dictionary {code: {unit: [values]}}
    stats = {}
    for (code, unit), values in grouped.items():
        if code not in stats:
            stats[code] = {}
        stats[code][unit] = values

    return num_df, stats


def tally_lab_values():
    """Creates statistic tables for numeric and nonnumeric laboratory test

    Tables created here are used for preprocessing laboratory test values, specifically, for
    tokenizing nonnumeric values and normalizing numeric values.
        Tables created:
            - 'Numeric stats': contains stats such as  mean values and standard deviations
                for each unique laboratory test-unit pair.
            - 'nonnumeric stats': contains frequencies of values and tokens to be mapped
    These maps are saved in 'config.DIR_VALUE_STATS'.
    """
    # Ensure no residual temporary files exist before steps.
    for pattern in [
        get_settings("TEMP_NONNUM_STATS_PTN"),
        get_settings("TEMP_NUM_STATS_PTN"),
    ]:
        residual_files = glob.glob(pattern)
        if residual_files:
            print("Residual temporary files found. Purging..")
            for file in residual_files:
                os.remove(file)
            print("Purged.")

    # General variables
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.LAB_RESULT_TABLE_PATTERN
    )
    chunksize = -1

    # ****************************
    # * Tally non-numeric values *
    # ****************************
    print("Collect non-numeric laboratory test results...")
    _ = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_tally_nonnums,
        chunksize=chunksize,
        pid_as_arg=True,
        patient_id_dict_as_arg=True,
    )
    print("Non-numerics collected.")
    print("Aggregating...")
    # Aggregate the intermediate nonnumeric stats tables created in step 1
    nonnum_stats_temp_paths = glob.glob(get_settings("TEMP_NONNUM_STATS_PTN"))
    temp_nonnum_tables = []
    for path in nonnum_stats_temp_paths:
        table = pd.read_pickle(path)
        temp_nonnum_tables.append(table)
        os.remove(path)
    nonnum_stats = pd.concat(temp_nonnum_tables)
    nonnum_stats = (
        nonnum_stats.groupby([config.COL_ITEM_CODE, config.COL_NONNUMERIC])
        .agg({"count": "sum"})
        .reset_index()
    )
    print("Tokenizing non-numerics...")
    # Apply the tokenizing function by laboratory test codes
    nonnum_stats = (
        nonnum_stats.groupby(config.COL_ITEM_CODE)
        .apply(_tokenize_nonnumerics)
        .reset_index(drop=True)
    )

    # Sort values
    nonnum_stats = nonnum_stats.sort_values(
        by=[config.COL_ITEM_CODE, "unique_count", "count"],
        ascending=[True, True, False],
    )

    # ************************
    # * Tally numeric values *
    # ************************
    print("Collect numeric test results...")
    tallied_nums = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_tally_nums,
        chunksize=chunksize,
        pid_as_arg=False,
        patient_id_dict_as_arg=True,
    )
    tallied_nums = tally_stats(tallied_nums)
    print("Non-numerics collected.")
    # Compute statistics
    print("Compute statistics...")
    final_stats = {}
    percentile_stats = {}
    for code, subdict in tallied_nums.items():
        if code not in final_stats:
            final_stats[code] = {}
        if code not in percentile_stats:
            percentile_stats[code] = {}
        for unit, values in subdict.items():
            if values:
                values = np.array(values, dtype=np.float32)

                # ***** Collect general stats (i.e, mean, std, etc.) *****
                # Trim outliers
                if len(values) >= 100 // config.OUTLIER_LIMIT:
                    lower_bound = np.percentile(values, config.OUTLIER_LIMIT)
                    upper_bound = np.percentile(values, 100 - config.OUTLIER_LIMIT)
                    clipped_vals = np.clip(values, lower_bound, upper_bound)
                else:
                    lower_bound = np.min(values)
                    upper_bound = np.max(values)
                    clipped_vals = values
                # Compute stats
                final_stats[code][unit] = {
                    "unit": unit,
                    "count": len(values),
                    "mean": np.mean(clipped_vals),
                    "med": np.median(clipped_vals),
                    "std": np.std(clipped_vals, ddof=0),
                    "mad": np.median(np.abs(clipped_vals - np.median(clipped_vals))),
                    "iqr": np.percentile(clipped_vals, 75)
                    - np.percentile(clipped_vals, 25),
                    "min": np.min(values),
                    "max": np.max(values),
                    "lower_limit": lower_bound,
                    "upper_limit": upper_bound,
                }

                # ***** Collect percentiles *****
                # NOTE: percentiles do not consider config.OUTLIER_LIMIT
                num_bins = get_settings("NUMERIC_BINS")
                percentile_steps = discretize_percentiles(num_bins=num_bins)

                # Compute percentiles
                percentile_values = np.percentile(
                    values, percentile_steps * 100, method="nearest"
                )
                # NOTE: percentile_values may contain duplicated values
                percentile_stats[code][unit] = percentile_values

    # Convert nested dictionary to DataFrame
    rows = []
    for code, subdict in final_stats.items():
        for unit, stats in subdict.items():
            row = {"code": code, "unit": unit, **stats}
            rows.append(row)
    flattened_percentiles = {
        (code, unit): values
        for code, units in percentile_stats.items()
        for unit, values in units.items()
    }

    # Clean the stats
    num_stats = pd.DataFrame(rows)
    num_stats.columns = [
        config.COL_ITEM_CODE,
        "unit",
        "count",
        "mean",
        "med",
        "std",
        "mad",
        "iqr",
        "min",
        "max",
        "lower_limit",
        "upper_limit",
    ]
    num_stats = num_stats.sort_values(
        [config.COL_ITEM_CODE, "count"], ascending=[True, False]
    )

    # Handling percentiles
    percentiles_df = pd.DataFrame.from_dict(flattened_percentiles, orient="index")
    percentiles_df.index = pd.MultiIndex.from_tuples(
        percentiles_df.index, names=[config.COL_ITEM_CODE, "unit"]
    )
    percentiles_df.reset_index(inplace=True)
    percentile_cols = percentile_steps.astype(str).tolist()
    percentiles_df.columns = [config.COL_ITEM_CODE, "unit"] + percentile_cols
    percentiles_df = pd.merge(
        percentiles_df,
        num_stats[[config.COL_ITEM_CODE, "unit", "count"]],
        on=[config.COL_ITEM_CODE, "unit"],
        how="left",
    )
    percentiles_df = percentiles_df[
        [config.COL_ITEM_CODE, "unit", "count"] + percentile_cols
    ]
    percentiles_df = percentiles_df.sort_values(
        [config.COL_ITEM_CODE, "count"], ascending=[True, False]
    )

    # **************
    # * Save stats *
    # **************
    num_stats.to_csv(
        get_settings("NUMERIC_STATS_PTN").replace("*", "train"),
        header=True,
        index=False,
    )

    nonnum_stats.to_csv(
        get_settings("NONNUMERIC_STATS_PTN").replace("*", "train"),
        header=True,
        index=False,
    )

    percentiles_df.to_csv(
        get_settings("NUMERIC_PERCENTILES_PTN").replace("*", "train"),
        header=True,
        index=False,
    )
