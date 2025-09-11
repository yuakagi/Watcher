"""Count unique codes in the dataset"""

import os
import glob
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import filter_records, parallel_map_partitions, map_code_to_name


def _count_unique_codes(
    df: pd.DataFrame, code_col: str, patient_id_dict: dict
) -> pd.DataFrame:
    """Counts unique code values from a table.

    Values are counted separately by patient IDs and periods.
    The product dataframe has following columns:
        config.COL_ORIGINAL_VALUE: unique codes.
        train ID and train period: value counts among train patient IDs and within train period
        train ID and test-validation period: value counts among train patient IDs and within test-validation period
        validation ID and train period: value counts among validation patient IDs and within train period
        validation ID and test-validation period: value counts among validation patient IDs and within test-validation period
        test ID and train period: value counts among test patient IDs and within train period
        test ID and test-validation period: value counts among test patient IDs and within test-validation period
    Args:
        df (pd.DataFrame): Dataframe to be processed.
        code_col (str): Name of the column that stores codes.
        patient_id_dict (dict): List of dictionaries that contain train, validation and train patient IDs.
            Each subdictionary looks like:
                {config.COL_PID: [patient_id, ...]}
    Returns:
        final_table (pd.DataFrame): Table that contains the full count results
    """
    final_table = None
    # Iterating through train, val, test patient IDs
    for flag, id_list in patient_id_dict.items():
        train_period_df = filter_records(df, id_list, period="train")
        test_period_df = filter_records(df, id_list, period="test")
        train_period_code_counts = (
            train_period_df[code_col].replace("", None).value_counts()
        )
        test_period_code_counts = (
            test_period_df[code_col].replace("", None).value_counts()
        )
        train_period_code_counts_df = pd.DataFrame(
            train_period_code_counts,
        )
        test_period_code_counts_df = pd.DataFrame(test_period_code_counts)
        train_period_code_counts_df.columns = [f"{flag}_ID_and_train_period"]
        test_period_code_counts_df.columns = [f"{flag}_ID_and_test_period"]
        temp_table = pd.merge(
            train_period_code_counts_df,
            test_period_code_counts_df,
            how="outer",
            left_index=True,
            right_index=True,
        )
        temp_table = temp_table.fillna(0)

        if final_table is None:
            final_table = temp_table
        else:
            final_table = pd.merge(
                final_table,
                temp_table,
                how="outer",
                left_index=True,
                right_index=True,
            )

    final_table = final_table.fillna(0)
    final_table = final_table.astype(int)
    final_table[config.COL_ORIGINAL_VALUE] = final_table.index
    final_table = final_table.reset_index(drop=True)
    final_table = final_table[
        [
            config.COL_ORIGINAL_VALUE,
            "train_ID_and_train_period",
            "train_ID_and_test_period",
            "validation_ID_and_train_period",
            "validation_ID_and_test_period",
            "test_ID_and_train_period",
            "test_ID_and_test_period",
        ]
    ]

    return final_table


def _count_dx_codes(df: pd.DataFrame, patient_id_dict: dict) -> pd.DataFrame:
    """Collects and counts diagnosis codes.

    Because diagnosis codes can be paired with provisional flags, diagnosis codes with provisional flags are treated as
    individual codes and they are counted separately.
    'config.PROV_SUFFIX' is appended at the ends of diagnosis codes with '1' in 'config.COL_PROVISIONAL_FLAG' here.
    Args:
        df (pd.DataFrame): Dataframe to be processed.
        patient_id_dict (dict): List of dictionaries that contain train, validation and train patient IDs.
    Returns:
        full_dx_code_count_table (dict): (pd.DataFrame): Table that contains the full count results
    """
    # Prepare dataframes separately with or without provisional flags
    prv_mask = df[config.COL_PROVISIONAL_FLAG] == 1
    df_without_so = df[~prv_mask]
    df_with_so = df[prv_mask]
    df_with_so.loc[:, config.COL_ITEM_CODE] = (
        df_with_so[config.COL_ITEM_CODE] + config.PROV_SUFFIX
    )

    # Count diagnosis codes withoutT provisional flags
    count_table = _count_unique_codes(
        df_without_so, config.COL_ITEM_CODE, patient_id_dict
    )

    # Count diagnosis codes with provisional flags
    so_count_table = _count_unique_codes(
        df_with_so, config.COL_ITEM_CODE, patient_id_dict
    )

    # Concatenate them
    full_dx_code_count_table = pd.concat([count_table, so_count_table])

    return full_dx_code_count_table


def _finalize_counts(df: pd.DataFrame, code_type: str) -> pd.DataFrame:
    """Finalizes value count tables.

    Because value counts are performed chunk by chunks and because these results are directly concatenated,
    value count tables must be loaded again for the final tally.
    Args:
        df (pd.DataFrame): Dataframe to be processed
        code_type (str): Type of code.
    Returns:
        df (pd.DataFrame): Finalized dataframe
    """
    cols = df.columns.tolist()
    for col in cols:
        if col != config.COL_ORIGINAL_VALUE:
            df[col] = df[col].astype(int)
    df = df.groupby(config.COL_ORIGINAL_VALUE).sum().reset_index()

    # Map codes to texts for readability
    df[config.COL_TEXT] = map_code_to_name(
        df[config.COL_ORIGINAL_VALUE], code_type=code_type
    )
    cols.remove(config.COL_ORIGINAL_VALUE)
    cols = [config.COL_ORIGINAL_VALUE, config.COL_TEXT] + cols
    df = df[cols]

    return df


def collect_unique_codes():
    """Collects standardized unique codes.

    These collected codes are the basis of the model's inference vocabulary.
    Because model's inference vocabulary is determined by the codes seen in the train dataset,
    codes are collected only from in-distribution records.
    The final product is a dictionary that contain multiple code lists. It has following sections:
        - 'diagnosis_code'
            - 'all': all unique diagnosis codes seen in the training data
            - 'with prov.': unique diagnosis codes seen at least once with a provisional flag
            - 'only with prov.': unique diagnosis codes that are not seen without a provisional flag
        - 'medication_code'
            - 'all': all unique medication codes seen in the training data
        - 'lab_test_code'
            - 'all': all unique laboratory test codes seen in the training data
    Args:
        None
    Returns:
        None
    """

    # Loop over coding systems (e.g, medication, diagnosis, etc.)
    # **********
    # * diagnosis *
    # **********
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.DIAGNOSIS_TABLE_PATTERN
    )

    _ = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_count_dx_codes,
        output_file_path=get_settings("DX_CODE_COUNTS_PTH"),
        chunksize=-1,
        single_file=True,
        patient_id_dict_as_arg=True,
    )

    source_path_pattern = get_settings("DX_CODE_COUNTS_PTH").replace(".csv", "*.csv")
    _ = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_finalize_counts,
        output_file_path=get_settings("DX_CODE_COUNTS_PTH"),
        chunksize=-1,
        single_file=True,
        code_type=config.DX_CODE,
    )

    # *******
    # * medication *
    # *******
    prescription_order_tables = glob.glob(
        os.path.join(
            get_settings("CLEANED_TABLES_DIR"), config.PRESCRIPTION_ORDER_TABLE_PATTERN
        )
    )
    injectable_order_tables = glob.glob(
        os.path.join(
            get_settings("CLEANED_TABLES_DIR"), config.INJECTION_ORDER_TABLE_PATTERN
        )
    )
    _ = parallel_map_partitions(
        source_path_pattern=prescription_order_tables + injectable_order_tables,
        function=_count_unique_codes,
        output_file_path=get_settings("MED_CODE_COUNTS_PTH"),
        chunksize=-1,
        single_file=True,
        patient_id_dict_as_arg=True,
        code_col=config.COL_ITEM_CODE,
    )

    source_path_pattern = get_settings("MED_CODE_COUNTS_PTH").replace(".csv", "*.csv")
    _ = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_finalize_counts,
        output_file_path=get_settings("MED_CODE_COUNTS_PTH"),
        chunksize=-1,
        single_file=True,
        code_type=config.MED_CODE,
    )

    # **********
    # * medication *
    # **********
    # Parse laboratory test result tables
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.LAB_RESULT_TABLE_PATTERN
    )
    _ = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_count_unique_codes,
        output_file_path=get_settings("LAB_CODE_COUNTS_PTH"),
        chunksize=-1,
        single_file=True,
        patient_id_dict_as_arg=True,
        code_col=config.COL_ITEM_CODE,
    )

    source_path_pattern = get_settings("LAB_CODE_COUNTS_PTH").replace(".csv", "*.csv")
    _ = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_finalize_counts,
        output_file_path=get_settings("LAB_CODE_COUNTS_PTH"),
        chunksize=-1,
        single_file=True,
        code_type=config.LAB_CODE,
    )
