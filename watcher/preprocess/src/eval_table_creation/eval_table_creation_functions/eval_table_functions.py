"""Functions to put labels"""

import os
import glob
import shutil
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import parallel_map_partitions, load_catalog, format_df


def _prepare_eval_tables(
    df: pd.DataFrame, file_name: str, temp_path_pattern: str, catalog: str
) -> pd.DataFrame:
    """Helper function to create tables for task performance evaluation.

    Args:
        df(pd.DataFrame): Target dataframe.
        file_name (str): Name of the source file.
        temp_path_pattern (str): Temporary file path pattern for saving the processed dataframe.
        catalog (str): Vocabulary catalog used to map categorical values to texts.
    Returns:
        df(pd.DataFrame): Processed dataframe with the new column of config.COL_LABEL.
            This is returned for consistency.
    """
    # Re-organize the data
    td_rows = df[config.COL_ORIGINAL_VALUE].fillna("") == ""
    df = df.drop(df[td_rows].index)
    df[config.COL_RESULT] = df[config.COL_ORIGINAL_VALUE].shift(-1, fill_value="")
    shifted_record_no = df[config.COL_RECORD_ID].shift(-1, fill_value="NO ID")
    same_origin = shifted_record_no == df[config.COL_RECORD_ID]
    df.loc[~same_origin, config.COL_RESULT] = ""
    df = df.drop(df[same_origin.shift(1, fill_value=False)].index)

    # Mapping texts
    df = pd.merge(df, catalog, on=config.COL_ORIGINAL_VALUE, how="left")

    # Organize columns
    df = df.rename(
        columns={
            config.COL_ORIGINAL_VALUE: config.COL_CODE,
            config.COL_PID: config.COL_PID,
            config.COL_TIMEDELTA: config.COL_AGE,
        }
    )

    # Dtype check and column selection
    df = format_df(df, table_params=config.EVAL_TABLE_COLS)

    # Save the file
    # NOTE: File is named in a way that its name matches the source aggregated table.
    file_name_parts = file_name.split("_")
    file_no = file_name_parts[-1].replace(".pkl", "")
    temp_path = temp_path_pattern.replace("*", file_no)
    df.to_pickle(temp_path)

    return df


def prepare_eval_tables():
    """Creates tables for task performance evaluation."""
    # Ensure that the temporary file directory is empty
    for old_file in os.listdir(get_settings("TEMP_DIR")):
        os.remove(os.path.join(get_settings("TEMP_DIR"), old_file))

    # Load catalog
    catalog = load_catalog(
        catalog_type="full", catalogs_dir=get_settings("CATALOGS_DIR")
    )
    catalog = catalog[[config.COL_ORIGINAL_VALUE, config.COL_TEXT]].astype(str)
    catalog = catalog.loc[catalog[config.COL_ORIGINAL_VALUE].fillna("") != ""]
    catalog = catalog.drop_duplicates(config.COL_ORIGINAL_VALUE)
    # Execute the main steps
    output_dir = get_settings("EVAL_TABLES_DIR")

    for flag in [config.TRAIN, config.VAL, config.TEST]:
        file_pattern = get_settings("FIN_AGGREGATED_FILE_PTN")
        source_path_pattern = file_pattern.replace("*", f"{flag}_*")
        temp_path_pattern = os.path.join(
            get_settings("TEMP_DIR"),
            config.EVAL_TABLE_FILE_PATTERN.replace("*", f"{flag}_*"),
        )

        _ = parallel_map_partitions(
            source_path_pattern=source_path_pattern,
            clear_temp_dir=False,  # <- Ensure not to delete temp files
            function=_prepare_eval_tables,
            chunksize=-1,  # <- Do not change the chunksize in this function
            file_name_as_arg=True,
            temp_path_pattern=temp_path_pattern,
            catalog=catalog,
        )

    # Move files
    for old_file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, old_file))
    temp_path_pattern = os.path.join(
        get_settings("TEMP_DIR"), config.EVAL_TABLE_FILE_PATTERN
    )
    for temp_file in glob.glob(temp_path_pattern):
        dst = os.path.join(output_dir, os.path.basename(temp_file))
        shutil.move(temp_file, dst)
