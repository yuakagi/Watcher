"""Module to aggregate records."""

import os
import shutil
import glob
from concurrent.futures import as_completed, ProcessPoolExecutor
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    parallel_map_partitions,
)


def _aggregate_records(df: pd.DataFrame, pid: int, patient_id_dict: dict):
    """Aggregates records of all different data types in the product tables.

        Tables are created by patient groups, therefore, this process ensures that
    records of a patient do not exist across different tables.
        The maximum number of patients stored in a table is defined by config.PATIENTS_PER_FILE.
        Because tables created here are further processed for finalization, all tables are saved
    in the 'get_settings("TEMP_DIR")' as pickled objects.
        Files are named 'aggregated_<process id>_<train/val/test>_<file number>.pkl'.
    Args:
        df (pd.DataFrame): Target dataframe.
        pid (int): Child process ID. This is passed from 'parallel_map_partitions'.
        patient_id_dict (dict): List of dictionaries with each subdictionary contain patient IDs.
            Patient ID dictionaries are stored in the order of train, validation and test. This list can be
            directly passed by 'parallel_map_partition' with 'patient_id_dict_as_arg = True'.
    """
    # Initialize variables
    drop_dup_subset = list(df.columns)
    drop_dup_subset.remove(config.COL_RECORD_ID)

    # Create a dataframe for data separation
    for flag, id_list in patient_id_dict.items():
        # Assign file numbers to patients
        temp_df = pd.DataFrame({config.COL_PID: id_list})
        temp_df["file_no"] = pd.Series(temp_df.index) // config.PATIENTS_PER_FILE
        df = pd.merge(left=df, right=temp_df, on=config.COL_PID, how="left")
        unique_file_numbers = list(df["file_no"].dropna().unique())

        # Add records to each file
        for file_no in unique_file_numbers:
            # Select records
            saved_records = df[df["file_no"] == file_no].drop("file_no", axis=1)
            temp_path = get_settings("TEMP_AGGREGATED_FILE_PTN").replace(
                "*", f"{pid}_{flag}_{str(int(file_no))}"
            )

            # Load existing temporary data and concatenate them
            if os.path.exists(temp_path):
                existing_record = pd.read_pickle(temp_path)
                saved_records = pd.concat([existing_record, saved_records])

            # Drop duplicated timedelta values
            # (Same timedelta values over different data types are NOT deleted.
            # For example, if some diagnosis codes are entered at timedelta X and some medications are orderd at timedelta X as well,
            # Two rows of the same timedelta X are preserved for each of them.)
            saved_records = saved_records.drop_duplicates(subset=drop_dup_subset)

            # Sort values
            saved_records = saved_records.sort_values(config.SORT_COLS)

            # Save
            saved_records.to_pickle(temp_path)

        # Drop 'file no' column
        df = df.drop("file_no", axis=1)


def _aggregate_files(file_identifier: str):
    """Helper function to aggregate tables created by _aggregate_records()"""
    # Initialize variables
    temp_file_pattern = get_settings("TEMP_AGGREGATED_FILE_PTN").replace(
        "*", f"*_{file_identifier}"
    )
    files = glob.glob(temp_file_pattern)

    # *********************************************
    # * Aggregate all tables saved chunk by chunk *
    # *********************************************
    final_table = None
    for file in files:
        temp_table = pd.read_pickle(file)
        if final_table is None:
            # Initialize the final table
            final_table = temp_table
            # Determine columns for detection of duplicates
            # NOTE: config.COL_RECORD_ID and config.COL_TIME_AVAILABLE are removed from the list to correctly detect duplicated timedelta rows.
            drop_dup_subset = list(final_table.columns)
            drop_dup_subset.remove(config.COL_RECORD_ID)
            drop_dup_subset.remove(config.COL_TIME_AVAILABLE)
        else:
            # Concatenate tables
            final_table = pd.concat([final_table, temp_table])
            # Drop duplicates
            final_table = final_table.drop_duplicates(subset=drop_dup_subset)
            # Sort values
            final_table = final_table.sort_values(config.SORT_COLS)
        # Remove temporary file
        os.remove(file)

    # ********
    # * Save *
    # ********
    file_name = os.path.basename(
        get_settings("AGGREGATED_FILE_PTN").replace("*", file_identifier)
    )
    output_file_path = os.path.join(get_settings("TEMP_DIR"), file_name)
    final_table.to_pickle(output_file_path)


def aggregate_records() -> None:
    """Aggregates records of different data types by patients.
    See '_aggregate_records' for details.
    """
    # Collect all files in the proccessed file dir
    source_files = glob.glob(
        os.path.join(get_settings("SEQUENCED_TABLES_DIR"), "*.pkl")
    )
    _ = parallel_map_partitions(
        source_path_pattern=source_files,
        function=_aggregate_records,
        chunksize=-1,
        single_file=False,
        pid_as_arg=True,
        patient_id_dict_as_arg=True,
        allow_debug=False,
    )
    # Collect all temporary files
    temp_files = glob.glob(get_settings("TEMP_AGGREGATED_FILE_PTN"))

    # Collect unique file identifiers
    file_identifiers = []
    for file in temp_files:
        file_name = os.path.basename(file)
        name_components = file_name.split("_")
        flag = name_components[-2]
        file_no = name_components[-1].replace(".pkl", "")
        file_identifier = f"{flag}_{file_no}"
        file_identifiers.append(file_identifier)
    file_identifiers = list(set(file_identifiers))

    # Execute parallelism
    max_workers = get_settings("MAX_WORKERS")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _aggregate_files,
                file_identifier=file_identifier,
            )
            for file_identifier in file_identifiers
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing tasks"
        ):
            _ = future.result()

    # Clear existing files
    aggregated_dir = get_settings("AGGREGATED_TABLES_DIR")
    for file in os.listdir(aggregated_dir):
        os.remove(os.path.join(aggregated_dir, file))

    # Move product files from the temp dir to the target dir
    for src in glob.glob(os.path.join(get_settings("TEMP_DIR"), "*.pkl")):
        dst = os.path.join(aggregated_dir, os.path.basename(src))
        shutil.move(src, dst)


def aggregate_records_single(tables: tuple[DataFrame]) -> DataFrame:
    """Aggregate records from a patient for model inference."""
    valid_tables = [d for d in tables if d is not None]
    agg_df = pd.concat(valid_tables)
    # Sort
    agg_df = agg_df.sort_values(config.SORT_COLS)
    # Drop duolicated timestamps
    ts_mask = agg_df["years"].notna()
    shifted_ts_mask = ts_mask.shift(1)
    double_ts = ts_mask & shifted_ts_mask
    agg_df = agg_df.loc[~double_ts]

    return agg_df
