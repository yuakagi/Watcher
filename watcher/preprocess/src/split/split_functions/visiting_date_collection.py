import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import format_df


def _collect_visiting_dates(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collects candidates for first and last visiting dates of patients."""
    df = pd.read_pickle(file_path)
    df = df[[config.COL_PID, config.COL_TIMESTAMP]]
    df = df.sort_values(
        by=[config.COL_PID, config.COL_TIMESTAMP], ascending=[True, True]
    )
    first_visits = df.drop_duplicates(subset=config.COL_PID, keep="first")
    last_visits = df.drop_duplicates(subset=config.COL_PID, keep="last")

    return first_visits, last_visits


def collect_visiting_dates():
    """Collects first and last visitng dates of patients."""
    max_workers = get_settings("MAX_WORKERS")
    cleaned_tables_dir = get_settings("CLEANED_TABLES_DIR")
    output_dir = get_settings("VISITING_DATES_DIR")
    output_table_path = os.path.join(output_dir, config.VISITING_DATE_TABLE)
    all_files = []
    # Collect all files
    all_files = []
    for cleaning_params in config.CLEANING_PARAMS.values():
        if config.COL_TIMESTAMP in cleaning_params["final_cols"]:
            table_name_pattern = cleaning_params["dst_table_pattern"]
            file_pattern = os.path.join(cleaned_tables_dir, table_name_pattern)
            all_files += glob.glob(file_pattern)

    # Initialize empty dataframes
    first_df = pd.DataFrame(
        {
            config.COL_PID: pd.to_datetime([]),
            config.COL_TIMESTAMP: pd.to_datetime([]),
        }
    )
    last_df = first_df.copy()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_collect_visiting_dates, file_path=f) for f in all_files
        ]

        for future in as_completed(futures):
            first_sub, last_sub = future.result()
            if first_sub.size and last_sub.size:
                # Concatenate the reuslts
                first_df = pd.concat([first_df, first_sub])
                last_df = pd.concat([last_df, last_sub])
                # Drop duplicated patient IDs
                first_df = first_df.sort_values(
                    by=[config.COL_PID, config.COL_TIMESTAMP],
                    ascending=[True, True],
                )
                last_df = last_df.sort_values(
                    by=[config.COL_PID, config.COL_TIMESTAMP],
                    ascending=[True, True],
                )
                first_df = first_df.drop_duplicates(subset=config.COL_PID, keep="first")
                last_df = last_df.drop_duplicates(subset=config.COL_PID, keep="last")

    # Finalize the dataframes
    first_df = first_df.rename(
        columns={config.COL_TIMESTAMP: config.COL_FIRST_VISIT_DATE}
    )
    last_df = last_df.rename(columns={config.COL_TIMESTAMP: config.COL_LAST_VISIT_DATE})
    visits_df = pd.merge(first_df, last_df, on=config.COL_PID)
    # Format
    visits_df = format_df(df=visits_df, table_params=config.VISITING_DATE_TABLE_COLS)
    # Save
    visits_df.to_pickle(output_table_path)
