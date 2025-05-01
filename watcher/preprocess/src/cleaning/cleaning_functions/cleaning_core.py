import os
import re
from datetime import datetime
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import psycopg
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    format_df,
    tally_stats,
    load_db_engine,
    load_psycopg_params,
)


def _validate_sql_param(param: str) -> bool:
    """Check a parameter for query to avoid SQL injection."""
    return re.match(r"^[a-zA-Z0-9_]+$", param)


def _get_records(
    patient_id: str,
    table: str,
    engine: Engine,
    start: str | None = None,
    end: str | None = None,
    schema: str = "public",
) -> pd.DataFrame:
    """Get records using a patient id
    Args:
        patient_id (str): Patient ID
        table (str): Table name
        engine (Engine): SQLalchemy engine.
        start (str): Timestamp of YYYY/mm/dd HH:MM format.
        end (str): Timestamp of YYYY/mm/dd HH:MM format.
        schema (str): Schema name.
    """
    # Validate params
    for p in [table, patient_id, schema]:
        if not _validate_sql_param(p):
            raise ValueError("Invalid SQL parameter.")

    # Make querys
    query = f"""SELECT * FROM {schema}.{table} WHERE {config.COL_PID} = %(pid)s"""
    params = {
        "pid": patient_id,
    }
    if table != config.TB_PATIENTS:
        if end is not None:
            end_ts = datetime.strptime(end, config.DATETIME_FORMAT)
            query += " AND timestamp <= %(end)s"
            params["end"] = end_ts
        if start is not None:
            start_ts = datetime.strptime(start, config.DATETIME_FORMAT)
            query += " AND timestamp >= %(start)s"
            params["start"] = start_ts

    # Execute query
    df = pd.read_sql(
        query,
        con=engine,
        params=params,
    )

    return df


def _convert_timestamp_to_timedelta(
    df: pd.DataFrame, timestamp_col: str, dob_map: pd.DataFrame | datetime
) -> pd.Series:
    """Converts timestamps into timedelta by subtracting patients' DOBs.
    Args:
        df (pd.DataFrame): Dataframe to be processed
        timestamp_col (str): Name of the columns that store timestamp values.
            Objects in this column must be datetime objects.
        dob_map (pd.DataFrame|datetime): Table with patient IDs and DOBs.
            Alternatively, a datetime object can be passed. In this case, all timestapms are simply
            subtract by this value.
    Returns:
        timedelta_series (pd.Series): Timedelta onbjects
    """
    # 1. For inference with one patient
    if isinstance(dob_map, datetime):
        timedelta_series = df[timestamp_col] - dob_map

    # 2. For preprocessing
    else:
        df = pd.merge(left=df, right=dob_map, on=config.COL_PID, how="left")
        df[config.COL_DOB] = pd.to_datetime(df[config.COL_DOB], format="%Y/%m/%d")
        timedelta_series = df[timestamp_col] - df[config.COL_DOB]
        df = df.drop(config.COL_DOB, axis=1)

    return timedelta_series


def _cleaning_core(
    df: pd.DataFrame, cleaning_params: dict, dob_map: pd.DataFrame | datetime
) -> tuple:
    """Performs data cleaning processes that are applied to all data tables.
        Cleaning steps:
            - 1. Drop and count missing values
            - 2. Create a column of timedelta objects from the timestamp column
                (Skipped if the table does not contain any timestamp records.)
            - 3. Select and reorder columns
                (Skipped if cleaning_params["final_cols"] is empty.)
    Args:
        df (pd.DataFrame): Dataframe to be cleaned.
        cleaning_params (dict): Parameters for data cleaning.
        dob_map (pd.DataFrame): Table containing a column of patient IDs and another column of DOBs.
            If the main dataframe only contains a single patient, then a datetime object can be passed.
    Returns:
        df (pd.DataFrame): Cleaned dataframe.
        stats (dict): Dictionary containing analytics of this process.
    """
    # Initialize stats
    stats = {}
    stats["records before cleaning"] = len(df)

    # Initialize the analytics
    # TODO: Refine analytics, consider for single patient usage. Not just count numbers, record unique values where possible.
    # ******************
    # * Missing values *
    # ******************
    missings_dict = {}
    dropna_any = cleaning_params["dropna_any"]
    if dropna_any:
        # Count missing values to be dropped
        for col in dropna_any:
            missings_mask = df[col].isna()
            n_missings = int(missings_mask.sum())
            missings_dict[f"missing values in {col}"] = n_missings
        missings_dict["total records with missing values (dropped)"] = int(
            df[dropna_any].isna().any(axis=1).sum()
        )
        # Drop records with missing gvalues
        df = df.dropna(subset=dropna_any, how="any").copy()

    dropna_all = cleaning_params["dropna_all"]
    # Drop records with values missing together
    if dropna_all:
        # Count missing values to be dropped
        col_list_str = "-".join(dropna_all)
        missings_dict[
            f"records with values missing together ({col_list_str}, dropped)"
        ] = int(df[dropna_all].isna().all(axis=1).sum())
        # Drop records with missing values
        df = df.dropna(subset=dropna_all, how="all")

    stats["excluded records for missing vlues"] = missings_dict.copy()

    # **************
    # * timestamps *
    # **************
    if config.COL_TIMESTAMP in df.columns:
        # Convert to datetime
        df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])
        # Floor timestamp col to the level of minutes
        df[config.COL_TIMESTAMP] = df[config.COL_TIMESTAMP].dt.floor(freq="min")
        # Create a column of timedelta objects from the timestamp col
        timedelta_series = _convert_timestamp_to_timedelta(
            df, config.COL_TIMESTAMP, dob_map
        )
        df[config.COL_TIMEDELTA] = timedelta_series.values
        # Drop records with negative timedelta values
        negative_tds = df[config.COL_TIMEDELTA].dt.total_seconds() < 0
        stats["records with negative age (excluded)"] = int(negative_tds.sum())
        df = df[~negative_tds]

    # *********************
    # * Add optional cols *
    # *********************
    # Col for timestamps of data availability (e.g., reporting time of lab tests)
    # NOTE: Missing values are replaced with zeros.
    if config.COL_TIME_AVAILABLE not in df.columns:
        df[config.COL_TIME_AVAILABLE] = pd.Timedelta(0, "m")
    else:
        # Convert to timedelta
        df[config.COL_TIME_AVAILABLE] = pd.to_datetime(df[config.COL_TIME_AVAILABLE])
        df[config.COL_TIME_AVAILABLE] = df[config.COL_TIME_AVAILABLE].dt.floor(
            freq="min"
        )
        timedelta_series = _convert_timestamp_to_timedelta(
            df, config.COL_TIME_AVAILABLE, dob_map
        )
        df[config.COL_TIME_AVAILABLE] = timedelta_series.values
        negative_tds = df[config.COL_TIME_AVAILABLE].dt.total_seconds() < 0
        df.loc[negative_tds, config.COL_TIME_AVAILABLE] = pd.Timedelta(0, "m")

    # All other cols
    for col in cleaning_params["final_cols"].keys():
        if col not in df.columns:
            df[col] = None

    # *********************************************
    # * Final datatype check and column selection *
    # *********************************************
    df = format_df(df=df, table_params=cleaning_params["final_cols"])

    # Finalize stats
    stats["records after cleaning"] = len(df)

    return df, stats


def _clean_via_postgres(
    table: str,
    schema: str,
    cleaning_params: dict,
    dob_map: pd.DataFrame,
    output_path: str,
    n_chunks: int = 10,
):
    """Select records from the selected patient IDs.
    Records are extracted using batched query.
    """
    # NOTE: the placeholder '%s' is used to avoid SQL injection.
    all_dfs = []
    patient_ids = dob_map[config.COL_PID].tolist()
    batch_size = math.ceil(len(patient_ids) / n_chunks)

    # Connect to db and execute query
    connect_param = load_psycopg_params()
    with psycopg.connect(connect_param) as conn:
        cursor = conn.cursor()
        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i : i + batch_size]
            query = f"SELECT * FROM {schema}.{table} WHERE patient_id = ANY(%s);"
            cursor.execute(query, (batch,))
            rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            batch_df = pd.DataFrame(rows, columns=column_names)
            all_dfs.append(batch_df)

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Clean
    df, stats = _cleaning_core(df, cleaning_params, dob_map)

    # Save
    df.to_pickle(output_path)

    return stats


def clean_records_for_dataset():
    """Cleans all clinical record tables."""
    # Set up
    max_workers = get_settings("MAX_WORKERS")
    schema = get_settings("DB_SCHEMA")
    chunk_size = get_settings("PATIENTS_PER_FILE")
    debug = get_settings("DEBUG_MODE")
    debug_chunks = get_settings("DEBUG_CHUNKS")
    all_substats = []

    # Validate params
    if not _validate_sql_param(schema):
        raise ValueError("Invalid SQL parameter passed.")

    # Load all patient IDs and DOBs
    engine = load_db_engine()
    full_dob_map = pd.read_sql(
        f"SELECT {config.COL_PID}, {config.COL_DOB} from {schema}.{config.TB_PATIENTS}",
        con=engine,
    )
    full_dob_map = full_dob_map.dropna(how="any")

    # Parallelism
    stats_list = []
    n_chunks = math.ceil(len(full_dob_map) / chunk_size)
    if debug:
        n_chunks = min(n_chunks, debug_chunks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for table, cleaning_params in config.CLEANING_PARAMS.items():
            record_type = cleaning_params["record_type"]
            path_pattern = os.path.join(
                get_settings("CLEANED_TABLES_DIR"), cleaning_params["dst_table_pattern"]
            )
            futures = [
                executor.submit(
                    _clean_via_postgres,
                    table=table,
                    schema=schema,
                    cleaning_params=cleaning_params,
                    dob_map=full_dob_map.iloc[i * chunk_size : (i + 1) * chunk_size],
                    output_path=path_pattern.replace("*", str(i)),
                )
                for i in range(0, n_chunks)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"cleaning {record_type}",
            ):
                stats = future.result()
                stats_list.append(stats)

            sub_stats = {record_type: tally_stats(stats_list)}
            all_substats.append(sub_stats)

    final_stats = tally_stats(all_substats)

    return final_stats


def clean_records_for_inference(
    patient_id: str,
    start: str,
    end: str,
    db_schema: str = "public",
) -> tuple[tuple[pd.DataFrame, ...], datetime]:
    """Helper function that cleans laboratory test results for a single patient."""
    # Create engine
    engine = load_db_engine()

    # Validate params
    if not _validate_sql_param(patient_id):
        raise ValueError("Invalid SQL parameter passed.")

    # Get DOB
    query = text(
        f"SELECT {config.COL_DOB} FROM {config.TB_PATIENTS} WHERE {config.COL_PID} = :patient_id"
    )

    with engine.connect() as conn:
        result = conn.execute(query, {"patient_id": patient_id}).fetchone()
    if result is None:
        # Raise an error if patient id is not found
        raise ValueError(f"Patient ID '{patient_id}' not found in database.")
    dob = result[0] if result else None

    # Convert to datetime (default time is midnight)
    dob = datetime.combine(dob, datetime.min.time())

    # Extract and clean records by tables
    cleaned_df_list = []
    for table, cleaning_params in config.CLEANING_PARAMS.items():
        df = _get_records(
            patient_id=patient_id,
            table=table,
            engine=engine,
            start=start,
            end=end,
            schema=db_schema,
        )
        df, _ = _cleaning_core(
            df=df,
            cleaning_params=cleaning_params,
            dob_map=dob,
        )
        cleaned_df_list.append(df)

    return tuple(cleaned_df_list), dob
