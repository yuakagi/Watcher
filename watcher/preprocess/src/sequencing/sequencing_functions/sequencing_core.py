"""Module to align laboratory test result records"""

import numpy as np
import pandas as pd
from .....general_params import watcher_config as config
from .....utils import preprocess_timedelta_df, format_df


def align_records_linearly(
    df: pd.DataFrame,
    task_no: int,
    categorical_dim: int,
    use_timestamps: bool = True,
    use_numeric: bool = True,
    use_nonnumeric: bool = True,
    use_categorical: bool = True,
):
    """Aligns timedelta records, numeric values and categorical values into distinct rows to create the basis of the patient timeline matrices.

    Columns are aligned by timedelta, values, codes.
    Args:
        df (pd.DataFrame): Dataframe to be processed.
        task_no (int): Task number.
        categorical_dim (int): Maximum number of embedding indexes to represent a code.
        use_timestamps (bool): If true, timedelta paired with records are processed.
        use_numeric (bool): If true, numeric values are processed, padded otherwise.
        use_nonnumeric (bool): If true, nonnumeric values are processed, padded otherwise.
        use_categorical (bool): If true, code values are processed, padded otherwise.
    Returns:
        organized_df (pd.DataFrame): Processed dataframe
    """
    # Define the padding values
    timedelta_pad = np.nan
    numeric_pad = np.nan
    categorical_pad = 0

    # Prepare and sort columns.
    item_numbers = np.arange(1, len(df) + 1, dtype=int)
    df[config.COL_PRIORITY] = 0
    df[config.COL_TASK_NO] = task_no
    df[config.COL_ITEM_NO] = item_numbers
    categorical_cols = [f"c{i}" for i in range(categorical_dim)]
    final_cols = (
        config.PREPROCESS_META_COLS
        + config.TIMEDELTA_COMPONENT_COLS
        + [config.COL_NUMERIC]
        + categorical_cols
    )
    target_cols = final_cols + [
        config.COL_NONNUMERIC,
        config.COL_ORIGINAL_NONNUMERIC,
        config.COL_ORIGINAL_NUMERIC,
    ]
    df = df.loc[:, target_cols].copy()

    process_df_list = []

    # ************************
    # * Timedelta components *
    # ************************
    # Initialize the product dataframe ('organized_df') and process timedelta.
    if use_timestamps:
        td_df = df.copy()
        # Empty config.COL_ORIGINAL_VALUE col
        td_df.loc[:, config.COL_ORIGINAL_VALUE] = None
        # Pad numeric and nonnumeric values
        td_df.loc[:, config.COL_NUMERIC] = numeric_pad
        # Pad categorical values
        td_df.loc[:, categorical_cols] = categorical_pad
        # Set the priority of timedelta values to zero to ensure that they are aligned ahead of other records
        td_df.loc[:, config.COL_PRIORITY] = 0
        # Fill task and item numbers with -1 so that duplicated timestamp rows from different chunk files can be removed esily later.
        td_df.loc[:, config.COL_TASK_NO] = -1
        td_df.loc[:, config.COL_ITEM_NO] = -1
        # Append to the list
        process_df_list.append(td_df)

    # **********************************
    # * Numeric and nonnumeric values *
    # **********************************
    # NOTE: Numeric values and nonnumeric values should NOT coexist in one row.
    if use_nonnumeric or use_numeric:
        v_df = df.copy()  # df for (non-)numeric values
        # Initialize config.COL_ORIGINAL_VALUE col
        v_df.loc[:, config.COL_ORIGINAL_VALUE] = v_df[config.COL_ORIGINAL_NUMERIC]
        # Pad timedelta
        v_df.loc[:, config.TIMEDELTA_COMPONENT_COLS] = timedelta_pad
        # Pad categorical
        v_df.loc[:, categorical_cols] = (
            categorical_pad  # v_df now only contains nonnumeric values.
        )

        if use_nonnumeric:
            # Ensure that numeric and nonnumeric values do not coexist.
            num_mask = ~(v_df[config.COL_NUMERIC].isna())
            v_df[config.COL_NONNUMERIC] = v_df[config.COL_NONNUMERIC].mask(
                num_mask, categorical_pad
            )
            # Fill NA
            v_df[config.COL_NONNUMERIC] = v_df[config.COL_NONNUMERIC].fillna(0)
            # Copy nonnumeric values to categorical col
            v_df["c0"] = v_df[config.COL_NONNUMERIC]
            # Replace values in the COL_ORIVINAL_VALUE
            v_df[config.COL_ORIGINAL_VALUE] = v_df[config.COL_ORIGINAL_VALUE].mask(
                ~num_mask, v_df[config.COL_ORIGINAL_NONNUMERIC]
            )
        # Set the priority of (non-)numeric values to 2, to ensure that they come after timestamps and
        #   other catgorical values (such aslaboratory test items).
        v_df[config.COL_PRIORITY] = 2
        # Append to the list
        process_df_list.append(v_df)

    # **********************************
    # * Categorical values             *
    # * (Except for nonnumeric values)*
    # **********************************

    # This process modifies the passed dataframe 'df' directly, and deletes it at the end, therefore this must come finally.
    if use_categorical:
        c_df = df.copy()
        # Pad timedelta
        c_df.loc[:, config.TIMEDELTA_COMPONENT_COLS] = timedelta_pad
        # Pad numeric and nonnumeric values
        c_df.loc[:, config.COL_NUMERIC] = (
            numeric_pad  # df now only contains coded data.
        )
        # Set the priority of categorivcal values except for nonnumeric values to 1 to ensure that they come after timedelta values and before (non-)numeric values
        c_df.loc[:, config.COL_PRIORITY] = 1
        # Append to the list
        process_df_list.append(c_df)

    # Organize columns for the final product
    organized_df = pd.concat(process_df_list)
    organized_df = organized_df[final_cols]

    # Drop duplicates. If this is not performed, duplicated timestamp tokens are left in the trjectories.
    drop_dup_subset = final_cols.copy()
    drop_dup_subset.remove(config.COL_RECORD_ID)
    organized_df = organized_df.drop_duplicates(subset=drop_dup_subset)

    # Drop rows with neither timedelta, categorical or numeric values
    organized_df[categorical_cols] = organized_df[categorical_cols].astype(int)
    empty_mask = (
        organized_df[config.COL_YEARS].isna()  # Empty timedelta
        & (organized_df["c0"] == categorical_pad)  # Empty categorical
        & organized_df[config.COL_NUMERIC].isna()  # Empty numeric
    )
    organized_df = organized_df[~empty_mask]

    # Alternate rows, sorting values by patient ID, timedelta, data type, task number, item number, priority
    organized_df = organized_df.sort_values(
        config.SORT_COLS,
        ascending=[True, True, True, True, True, True],
    )

    return organized_df


def sequence_core(
    df: pd.DataFrame,
    task_no: int,
    categorical_dim: int,
    type_no: int,
    dropna_subset: list,
) -> pd.DataFrame:
    """Processes the common cequencing steps."""
    stats = {}
    # **************
    # * Initialize *
    # **************
    # Check value use
    use_categorical = "c0" in df.columns
    use_numeric = config.COL_NUMERIC in df.columns
    use_nonnumeric = config.COL_NONNUMERIC in df.columns
    # Padding columns
    categorical_cols = [f"c{i}" for i in range(categorical_dim)]
    if config.COL_ORIGINAL_VALUE not in df.columns:
        df[config.COL_ORIGINAL_VALUE] = ""
    if config.COL_ORIGINAL_NUMERIC not in df.columns:
        df[config.COL_ORIGINAL_NUMERIC] = ""
    if config.COL_ORIGINAL_NONNUMERIC not in df.columns:
        df[config.COL_ORIGINAL_NONNUMERIC] = ""
    for c in categorical_cols:
        if c not in df.columns:
            df[c] = 0
    if not use_numeric:
        df[config.COL_NUMERIC] = pd.Series([], dtype=float)
    if not use_nonnumeric:
        df[config.COL_NONNUMERIC] = pd.Series([], dtype=float)
    # Set record type number
    df[config.COL_TYPE] = type_no

    # ***********************
    # * Drop missing values *
    # ***********************
    n_before_dropna = len(df)
    stats["missing values"] = {}
    for c in dropna_subset:
        n_missing = df[c].isna().sum()
        stats["missing values"][c] = n_missing
    df = df.dropna(subset=dropna_subset, how="any")
    stats["missing values"]["total"] = n_before_dropna - len(df)

    # *****************
    # * OOV handlings *
    # *****************
    n_oovs = 0
    # Handling categorical OOV (process categorical values first to avoid overlapping counts with nonnumerics.)
    if use_categorical:
        cat_oov = df["c0"] == config.OOV_INDEX
        if cat_oov.any():
            # Count OOV
            cat_oov_counts = (
                df.loc[cat_oov, config.COL_ORIGINAL_VALUE].value_counts().to_dict()
            )
            stats["out-of-vocabulary categorical values"] = cat_oov_counts
            n_oovs += cat_oov.sum()
            # Drop rows with OOV categorical values
            df = df[~cat_oov]
    # Handling nonnumeric OOV
    if use_nonnumeric:
        nonnum_oov = df[config.COL_NONNUMERIC] == config.OOV_INDEX
        if nonnum_oov.any():
            # Count
            df["nonnum_cat_pair"] = (
                df[config.COL_ORIGINAL_NONNUMERIC]
                + " ("
                + df[config.COL_ORIGINAL_VALUE]
                + ")"
            )
            nn_oov_counts = (
                df.loc[nonnum_oov, "nonnum_cat_pair"].value_counts().to_dict()
            )
            stats["out-of-vocabulary nonnumeric values"] = nn_oov_counts
            df = df.drop("nonnum_cat_pair", axis=1)
            n_oovs += nonnum_oov.sum()
            # Drop rows with OOV nonnumeric values
            df = df[~nonnum_oov]
    # Record
    stats["total out-of-vocabulary values"] = n_oovs

    # *******************
    # * Main sequencing *
    # *******************
    # Process timestamps
    df = preprocess_timedelta_df(df)
    # Preprocessing for sequencing.
    df = align_records_linearly(
        df,
        task_no,
        categorical_dim,
        use_categorical=use_categorical,
        use_nonnumeric=use_nonnumeric,
        use_numeric=use_nonnumeric,
    )
    # Dtype checking and column selection
    df = format_df(df, table_params=config.SEQUENCED_TABLE_COLS)

    return df, stats
