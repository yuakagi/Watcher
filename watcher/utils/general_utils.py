"""Utils designed for general purposes"""

import os
import sys
import traceback
import atexit
import glob
import shutil
import json
import subprocess
import re
import time
from datetime import datetime
from urllib.parse import quote_plus
from typing import Iterator, Callable, Any, Literal
from multiprocessing import Manager, current_process, Process
from concurrent.futures import as_completed, ProcessPoolExecutor
from torch import nn
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from tqdm import tqdm
from ..general_params import watcher_config as config
from ..general_params import get_settings


def load_db_params() -> dict:
    """Gets Postgres parameters from the environment."""
    db_params = {
        "db_user": os.environ.get("POSTGRES_USER"),
        "db_password": os.environ.get("POSTGRES_PASSWORD"),
        "db_name": os.environ.get("POSTGRES_DB"),
        "db_host": "db",
        "db_port": 5432,
    }
    return db_params


def load_psycopg_params():
    """Creates params for psycopg database connection"""
    db_params = load_db_params()
    # DO NOT use quote_plus for psycopg — it's for URL strings, not plain text
    psycopg_param = (
        f"dbname={db_params['db_name']} "
        f"user={db_params['db_user']} "
        f"password={db_params['db_password']} "
        f"host={db_params['db_host']} "
        f"port={db_params['db_port']}"
    )
    return psycopg_param


def load_db_engine() -> Engine:
    """Creates an engine for sqlalchemy"""
    db_params = load_db_params()
    engine_params = "postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}".format(
        db_user=db_params["db_user"],
        db_password=quote_plus(db_params["db_password"]),
        db_host=db_params["db_host"],
        db_port=db_params["db_port"],
        db_name=db_params["db_name"],
    )
    engine = create_engine(engine_params)
    return engine


class LogRedirector:
    """Context manager to redirect standard output and error logs to a text file."""

    def __init__(
        self, log_dir: str = None, file_name: str = None, create_dirs: bool = True
    ) -> None:
        """
        Initializes the LogRedirector with an optional log file path.

        Args:
            log_dir (str, optional): Path to the directory where the log file will be saved.
                If None, no redirection occurs.
            file_name (str, optional): Name of the log file.
                If not specified, the file name will be 'logfile_<current time>.txt'.
            create_dirs (bool, optional): If true, directories are automatically created.
        """
        self.redirect = log_dir is not None
        if self.redirect:
            if file_name is None:
                # Generate a file name based on the current time if not provided
                dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
                file_name = f"logfile_{dt_str}.txt"
            # If file_name is an absolute path, skip joining it with log_dir
            if not os.path.isabs(file_name):
                log_file = os.path.join(log_dir, file_name)
            else:
                log_file = file_name

            self.log_file = os.path.abspath(log_file)
            # Check path
            abs_log_dir = os.path.dirname(self.log_file)
            if create_dirs:
                os.makedirs(abs_log_dir, exist_ok=True)
            else:
                if not os.path.exists(abs_log_dir):
                    raise ValueError(f"Directory ({abs_log_dir}) does not exist.")
        else:
            self.log_file = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.writer = None

    def __enter__(self):
        """Enters the context manager, redirecting stdout and stderr to the log file."""
        if self.redirect:
            try:
                # Instantiate a writer
                self.writer = open(self.log_file, "w", encoding="utf-8")
                sys.stdout = self.writer
                sys.stderr = self.writer
                # Register a clean-up function that is executed even when the process abnormally finishes inside the context manager.
                atexit.register(
                    _cleanup_log_redirects,
                    {
                        "original_stdout": self.original_stdout,
                        "original_stderr": self.original_stderr,
                        "writer": self.writer,
                    },
                )
                print(f"Standard outputs and errors are redirected to {self.log_file}")
                print("See the file for details.")

            except Exception as e:
                # Restore the original stdout and stderr in case of failure
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
                print(f"Failed to redirect logs: {e}")
                raise

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exits the context manager, restoring stdout and stderr.

        Args:
            exc_type, exc_value, exc_traceback: Exception information if raised during the block.
        """
        if self.redirect:
            if exc_type is not None:
                traceback.print_exception(exc_type, exc_value, exc_traceback)
                self.writer.flush()
            _cleanup_log_redirects(
                self.original_stdout, self.original_stderr, self.writer
            )
            atexit.unregister(_cleanup_log_redirects)


def _cleanup_log_redirects(original_stdout, original_stderr, writer):
    """Cleans up log redirects."""
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    if writer:
        writer.close()


def count_model_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Counts parameters of a Pytorch's nn.Module instance.
    Args:
        model (nn.Module): nn.Module instance.
        trainable_only (bool): If true, the returned value only includes trainable parameters.
    Returns:
        total_n_params (int): Total number of parameters of the model.
    """
    if trainable_only:
        params = [param.numel() for param in model.parameters() if param.requires_grad]
    else:
        params = [param.numel() for param in model.parameters()]
    total_n_params = sum(params)

    return total_n_params


def load_patient_id_dict(included_only: bool = False) -> dict:
    """Loads patient ID lists for training, validation and test.
    Args:
        included_only (bool): If true, lists of patient IDs that are included for study. Otherwise full patient IDs are returned.
            Note that the list of included patient IDs are created upon the data labelling process.
    Returns:
        id_dict (dict): Dictionary with the keys of 'train', 'validation' and 'test', and corresponding ID lists are stored.
    """
    # Config
    id_list_dir = get_settings("PID_DIR")
    if included_only:
        file_name = config.INCLUDED_PATIENT_ID_LIST
    else:
        file_name = config.TOTAL_PATIENT_ID_LIST

    # Load
    file_path = os.path.join(id_list_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        id_dict = json.load(f)

    return id_dict


def filter_records(
    df: pd.DataFrame, patient_id_list: list, period: Literal["train", "val", "test"]
) -> pd.DataFrame:
    """Filters records by both patient IDs and periods.
    Args:
        df (pd.DataFrame): Dataframe to be filtered.
        patient_id_list (list): List of IDs for the patients to be included.
        train_period (bool): If true, records of train periods are selected.
    Returns:
        df (pd.DataFrame): Filtered dataframe.
    """
    # ****************
    # * Filter by ID *
    # ****************
    id_df = pd.DataFrame({config.COL_PID: patient_id_list})
    df = pd.merge(left=df, right=id_df, on=config.COL_PID, how="inner")

    # ***********************
    # * Filter by timestamp *
    # ***********************
    # Determine periods
    if period == "train":
        period_start = get_settings("TRAIN_PERIOD_START")
        period_end = get_settings("TRAIN_PERIOD_END")
    elif period == "test":
        period_start = get_settings("TEST_PERIOD_START")
        period_end = get_settings("TEST_PERIOD_END")
    else:
        raise ValueError("Invalid value for 'period' argument.")

    # Filter
    period_mask = df[config.COL_TIMESTAMP].between(period_start, period_end)
    df = df[period_mask]

    return df


def test_generator(generator: Iterator) -> Iterator:
    """Truncates a generator for testing."""
    n_iter = 0
    exhausted = False
    debug_chunks = get_settings("DEBUG_CHUNKS")
    while not exhausted:
        try:
            chunk = next(generator)
            yield chunk
            n_iter += 1
            if n_iter >= debug_chunks:
                exhausted = True
        except StopIteration:
            exhausted = True


def _parallel_map_partitions_wrapper_function(
    function,
    chunk: pd.DataFrame | str,
    temp_file_path: str,
    lock,
    shared_file_no,
    dtype: type,
    single_file: bool,
    file_name: str,
    pid_as_arg: bool,
    task_no_as_arg: bool,
    patient_id_dict_as_arg: bool,
    return_processed_df: bool,
    task_no: int,
    patient_id_dict: dict,
    header: bool,
    index: bool,
    **kwargs,
) -> dict | pd.DataFrame:
    """Helper function for 'parallel_map_partitions'.
    This function is passed to child processes, and each process executes this.
    See 'parallel_map_partitions' for details.
    """
    # Get the child process ID
    process = current_process()
    pid = process.pid

    # Check arguments
    if pid_as_arg:
        kwargs = {"pid": pid, **kwargs}
    if task_no_as_arg:
        kwargs = {"task_no": task_no, **kwargs}
    if patient_id_dict_as_arg:
        kwargs = {"patient_id_dict": patient_id_dict, **kwargs}
    if file_name is not None:
        kwargs = {"file_name": file_name, **kwargs}

    # Load dataframe
    if isinstance(chunk, str):
        if chunk.endswith("csv"):
            df = pd.read_csv(chunk, dtype=dtype, na_values=config.NA_VALUES, header=0)
        elif chunk.endswith("pkl"):
            df = pd.read_pickle(chunk)
        else:
            raise ValueError("Unknown format for DataFrame parsing:", chunk)
    else:
        df = chunk

    # Execute the given custom function
    result = function(df, **kwargs)

    # If the result is a tuple, extract df and stats. Otherwise, only df is returned.
    if isinstance(result, tuple) and len(result) == 2:
        df, stats = result
    else:
        df = result
        stats = None

    # Save
    if temp_file_path is not None:
        # Check file format
        if temp_file_path.endswith("csv"):
            extension = "csv"
        elif temp_file_path.endswith("pkl"):
            extension = "pkl"
        else:
            raise ValueError(
                "Unknown format passed. Files must be either csv or pkl.",
                temp_file_path,
            )
        # Aggregating files into a single file
        if single_file:
            with lock:
                # Append
                if os.path.exists(temp_file_path):
                    if extension == "csv":
                        df.to_csv(temp_file_path, mode="a", header=False, index=index)
                    else:
                        existing_df = pd.read_pickle(temp_file_path)
                        df = pd.concat([existing_df, df])
                        df.to_pickle(temp_file_path)
                # Save new
                else:
                    if extension == "csv":
                        df.to_csv(temp_file_path, mode="w", header=header, index=index)
                    elif extension == "pkl":
                        df.to_pickle(temp_file_path)

        # Saving files by chunks
        else:
            with lock:
                file_no = str(shared_file_no.value)
                shared_file_no.value += 1
            if "*" in temp_file_path:
                temp_file_path = temp_file_path.replace("*", file_no)
            else:
                temp_file_path = temp_file_path.replace(
                    f".{extension}", f"_{file_no}.{extension}"
                )
            if extension == "csv":
                df.to_csv(temp_file_path, mode="w", header=header, index=index)
            elif extension == "pkl":
                df.to_pickle(temp_file_path)
    else:
        # If output_file_path is not given, nothing is saved by this function.
        # Only execution of the given custom function is performed.
        pass

    # Return the processed dataframe if indicated
    if return_processed_df:
        return df

    # Otherwise, returns a dictionary
    if stats is not None:
        return stats


def parallel_map_partitions(
    source_path_pattern: str | list[str],
    function: Callable,
    chunksize: int,
    output_file_path: str = None,
    single_file: bool = True,
    file_name_as_arg: bool = False,
    pid_as_arg: bool = False,
    patient_id_dict_as_arg: bool = False,
    task_no_as_arg: bool = False,
    header: bool = True,
    index: bool = False,
    dtype: type = str,
    clear_temp_dir: bool = True,
    return_processed_df: bool = False,
    allow_debug: bool = True,
    **kwargs,
) -> list:
    """Maps a custom function to chunks of a table in parallel.
    A custom function must return a dataframe, and it can also return an additional output.

    Args:
        source_path_pattern (str): Target tables are searched with this pattern.
            Incude wild cards '*' in it properly. Files are searched by 'glob.glob()'.
            If '**/' is included in the pattern, 'glob.glob(recursive=True)' is applied.
            Alternatively, you can pass a list of paths.
            Files must be either csv or pkl, with proper file extensions (.csv or .pkl). Other form of files are
            not accepted.
        function (Callable): Custom function that is to be mapped to the table.
            This function is expected to take a pd.DataFrame as its first arugument and returns
            a pd.DataFrame with or without an additional dictionary.
        chunksize (int): Each table is loaded chunk by chunk with this chunk size.
            If chunksize is -1, then the entire table in a file is loaded by child processes.
            This is only applicable to csv files.
        output_file_path (str): Output table is saved in this path. Make sure not to include wilde cards ('*') in it.
        single_file (bool): If true, all tables processed at child processes are aggregated and saved as a single file.
        file_name_as_arg (bool): If true, the source file name is passed as an argument to the custom function.
        pid_as_arg (bool): If true, a child process ID (pid) is passed as an argument to the custom function.
        task_no_as_arg (bool): If true, the task number is passed as an argument.
        patient_id_dict_as_arg (bool): If true, a dictionary of patient IDs is passed as an argument.
        header (bool): If true, the processed tables are saved with header.
        index (bool): If true, the indexes of processed tables are also saved as a column in the csv files.
        dtype (type): This argument is directly passed to 'pd.read_csv()'.
        clear_temp_dir (bool): If true, the temporary file directory (temp_dir) is cleared.
            Set this to Flase if you use files remaining in the temporary directory in later steps.
        return_processed_df (bool): If true, processed dataframe objects are returned instead of a list of dictionaries.
        allow_debug (bool): If true, running on a debug mode is allowed. Otherwise, this function ignores the environment variable 'DEBUG_MODE'.
    Returns:
        stats_list (list of dictionaries): List of dictionaries that store statistics.
            If return_processed_df is true, then a list of dataframes are returned.
    """
    # Define variables.
    max_workers = get_settings("MAX_WORKERS")
    temp_dir = get_settings("TEMP_DIR")
    temp_file_name = os.path.basename(output_file_path) if output_file_path else None
    temp_file_path = (
        os.path.join(temp_dir, temp_file_name) if output_file_path else None
    )

    # Clear the temporary file directory
    if clear_temp_dir:
        for file in os.listdir(temp_dir):
            path = os.path.join(temp_dir, file)
            os.remove(path)
    # Load all patient ID lists if necessary
    if patient_id_dict_as_arg:
        full_id_dict = load_patient_id_dict(included_only=False)
    else:
        full_id_dict = {}

    # Create a generator that yields dataframe chunks.
    def _dataframe_generator(
        source_path_pattern: str, dtype: type, chunksize: int
    ) -> Iterator:
        """
        Yields:
            chunk (pd.dataframe|str): Dataframe chunk to be processed by a child process.
                If chunksize is -1, a path to a table is yielded instead so that
                the table can be loaded by a child process on its own.
                If the file is serialized, then chunksize is ignored.
        """
        if isinstance(source_path_pattern, list):
            paths = source_path_pattern
        elif "**" in source_path_pattern:
            paths = glob.glob(source_path_pattern, recursive=True)
        else:
            paths = glob.glob(source_path_pattern)

        for path in paths:
            # Check extension
            if path.endswith("csv"):
                extension = "csv"
            else:
                extension = "pkl"
            # Get the file name
            file_name = os.path.basename(path)
            if (chunksize == -1) or extension != "csv":
                # Yield a file path so that a whole table can be loaded by child processes
                chunk = path
                if file_name_as_arg:
                    yield chunk, file_name
                else:
                    yield chunk, None
            else:
                # A table is broken down into chunks
                df_reader = pd.read_csv(
                    path,
                    dtype=dtype,
                    chunksize=chunksize,
                    na_values=config.NA_VALUES,
                )
                for chunk in df_reader:
                    if file_name_as_arg:
                        yield chunk, file_name
                    else:
                        yield chunk, None

    chunk_iterator = _dataframe_generator(source_path_pattern, dtype, chunksize)

    # Truncate iterator for testing
    if get_settings("DEBUG_MODE") and allow_debug:
        chunk_iterator = test_generator(chunk_iterator)

    # Map functions to partition dataframes.
    stats_list = []
    # Execute parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor, Manager() as manager:
        lock = manager.Lock()
        shared_file_no = manager.Value("i", 0)

        futures = [
            executor.submit(
                _parallel_map_partitions_wrapper_function,
                function=function,
                chunk=chunk,
                temp_file_path=temp_file_path,
                single_file=single_file,
                lock=lock,
                shared_file_no=shared_file_no,
                dtype=dtype,
                header=header,
                index=index,
                file_name=file_name,
                pid_as_arg=pid_as_arg,
                task_no_as_arg=task_no_as_arg,
                patient_id_dict_as_arg=patient_id_dict_as_arg,
                task_no=task_no,
                return_processed_df=return_processed_df,
                patient_id_dict=full_id_dict,
                **kwargs,
            )
            for task_no, (chunk, file_name) in enumerate(chunk_iterator)
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing tasks"
        ):
            stats = future.result()
            if stats is not None:
                stats_list.append(stats)

    if output_file_path is not None:
        # Clear old files
        if "*" in output_file_path:
            old_file_pattern = output_file_path
        else:
            old_file_pattern = re.sub(r"\.(\w+)$", r"*.\1", output_file_path)
        old_files = glob.glob(old_file_pattern)
        for old_file in old_files:
            os.remove(old_file)
        # Move final products pylint: disable=unsupported-membership-test
        if "*" in temp_file_path:
            created_files = glob.glob(temp_file_path)
        else:
            created_files = glob.glob(re.sub(r"\.(\w+)$", r"*.\1", temp_file_path))
        dst_dir = os.path.dirname(output_file_path)
        for f in created_files:
            temp_file_name = os.path.basename(f)
            dst = os.path.join(dst_dir, temp_file_name)
            shutil.move(f, dst)

    return stats_list


def _pythonize(data: Any) -> Any:
    """Converts data into Python's built-in data types.
    The primary purpose of this fucntion is to convert nested list or dictionary objects
    so that they can be serialized as json.
    Array-like or dictionary-like objects created by pandas operations are often numpy objects, which are
    not json serializable. This function convert numpy objects back to Python's built-in objects.
    Args:
        data (Any): Data to be converted
    Returns:
        data (Any): Converted data
    """
    if isinstance(data, dict):
        return {key: _pythonize(value) for key, value in data.items()}

    if isinstance(data, list):
        return [_pythonize(item) for item in data]

    if isinstance(data, np.integer):
        return int(data)

    if isinstance(data, np.floating):
        return float(data)

    if isinstance(data, np.ndarray):
        return data.tolist()

    else:
        return data


def tally_stats(stats_list: list, pythonize_values: bool = True) -> dict:
    """Tallies stats in dictionaries and make them into a single dictionary.

    This function is supposed to take a list of dictionary returned by
    'parallel_map_partitions' when indicated.
    """

    def _tally_stats(d, final_d):
        for key, val in d.items():
            if isinstance(val, (int, float, np.number)):
                final_d[key] = final_d.get(key, 0) + val

            elif isinstance(val, list):
                final_d[key] = final_d.get(key, []) + val

            elif isinstance(val, dict):
                if key not in final_d:
                    final_d[key] = {}
                final_d[key] = _tally_stats(val, final_d[key])

        return final_d

    final_stats = {}
    for stats in stats_list:
        final_stats = _tally_stats(stats, final_stats)

    if pythonize_values:
        final_stats = _pythonize(final_stats)

    return final_stats


def load_catalog_info(catalogs_dir: str) -> dict:
    """Loads catalog info"""
    info_path = os.path.join(catalogs_dir, config.CATALOG_INFO_FILE)
    with open(info_path, "r", encoding="utf-8") as f:
        catalog_info = json.load(f)
    return catalog_info


CatalogTypes = Literal[
    "full",
    "all_special_tokens",
    "sex_tokens",
    "discharge_disposition_tokens",
    "operational_special_tokens",
    "nonnumeric_lab_values",
    "numeric_lab_values",
    "timedelta",
    config.DX_CODE,
    config.MED_CODE,
    config.LAB_CODE,
    "all_codes",
    "all_categorical",
]


def get_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Lists all the column names for categorical token indexes."""
    categorical_cols = df.filter(regex=r"c\d+").columns.tolist()
    return categorical_cols


def load_catalog(
    catalog_type: CatalogTypes = "full", catalogs_dir: str = None
) -> pd.DataFrame:
    """Loads Model's vocabulary catalogs.
    Args:
        catalog_type (str, optional): Type of catalog to be loaded.
        catalogs_dir (str, optional): Absolute path to the directory where the catalog and catalog info exist.
            If not specified, this function tries to load the catalog from 'CATALOGS_DIR'.
    Returns:
        catalog (pd.DataFrame): Loaded catalog.
            This catalog is indexed by the label indexes, which are exactly the same values with 'config.COL_LABEL' column.
    """
    # Load table
    if catalogs_dir is None:
        catalogs_dir = get_settings("CATALOGS_DIR")
    catalog_path = os.path.join(catalogs_dir, config.CATALOG_FILE)
    catalog = pd.read_csv(catalog_path, header=0, dtype=str, na_values=config.NA_VALUES)
    # Slice out catalog
    if catalog_type != "full":
        # Load the catalog info first
        catalog_info = load_catalog_info(catalogs_dir)
        min_idx = catalog_info["min_indexes"][catalog_type]
        max_idx = catalog_info["max_indexes"][catalog_type]
        # Slice
        catalog = catalog.loc[min_idx:max_idx, :]

    # Convert values to integers
    categorical_cols = get_categorical_cols(catalog)
    int_cols = categorical_cols + [config.COL_LABEL]
    catalog[int_cols] = catalog[int_cols].astype(int)

    # Ensure the catalog is indexed by the labels
    if not (catalog.index == catalog[config.COL_LABEL]).all():
        raise RuntimeError("Catalog index mismatch")
    else:
        pass

    return catalog


def load_special_token_index(special_token: str | list[str]) -> int | list[int]:
    """Returns the embedding index of a specific token.
    Args:
        special_token(str|list): Special token of interest.
            A list of strings can be passed alternatively if multiple indexes are needed.
    Returns:
        embed_idx(int|list): Embedding index of the special_token.
            If 'special_token' is a list, a list of integers are returned.
    """
    token_ref_path = get_settings("TOKEN_REFERENCE_PTH")
    with open(token_ref_path, "r", encoding="utf-8") as f:
        token_reference = json.load(f)
    special_token_map = token_reference["tokens_and_indexes"]["special_tokens"]

    if isinstance(special_token, list):
        embed_idx_list = []
        for token in special_token:
            embed_idx = special_token_map[config.COL_TOKEN].index(token)
            embed_idx_list.append(embed_idx)
        return embed_idx_list

    embed_idx = special_token_map[config.COL_TOKEN].index(special_token)
    return embed_idx


def load_categorical_dim() -> int:
    """Loads the maxmum number of embedding indexes to represent a code value.

    The value returnes by this is one of Watcher's hyperparameter (categorical_dim).
    Args:
        None
    Returns:
        categorical_dim(int): Maxmum number of embedding indexes to represent a code.
    """
    return get_settings("CATEGORICAL_DIM")


def load_special_token_dict() -> dict:
    """Creates a dictionary to map special tokens to embedding indexes.

    Args:
        None
    Returns:
        special_token_dict (dict): Dictionary that contain special tokens as keys and corresponding embedding indexes as values.
    """
    token_ref_path = get_settings("TOKEN_REFERENCE_PTH")
    with open(token_ref_path, "r", encoding="utf-8") as f:
        token_reference = json.load(f)
    special_token_dict = token_reference["tokens_and_indexes"]["special_tokens"]
    special_tokens = special_token_dict[config.COL_TOKEN]
    special_token_indexes = special_token_dict["index"]
    special_token_dict = {
        key: val for key, val in zip(special_tokens, special_token_indexes)
    }

    return special_token_dict


def load_model_params_from_dataset(info_path: str = None) -> dict:
    """Loads model the hyperparameters that are determined by the dataset.
    Args:
        info_path (str, optional): Path to the dataset info file (json).
            If not given, this function tries to load the path from the environment.
    Returns:
        hyperparameters (dict): Dictionary containing model's hyperparameter.
    """
    if info_path is None:
        info_path = get_settings("DATASET_INFO_PTH")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    hyperparameters = info["hyperparameters"]
    return hyperparameters


def flag_admission_status(
    df: pd.DataFrame, ignore_truncated: bool = False
) -> pd.Series:
    """Puts binary admission status flags (0 or 1) to the records in the labelled tables.
    Args:
        df (pd.DataFrame): Labelled table.
        ignore_truncated (bool): If true, truncated admission histories (missing either [ADM] or [DSC])
            are not counted as admitted.
            For the training, it should be set to false.
            In case you want to collect statistics on admissions and discharges (such as length of stay),
            consider setting this argument to false.
    Returns:
        binary_admission_status (pd.Series): Series of binary flags for admission status.
            Records within an admission is flagged with 1.
    """
    # Add a column for admission status
    dmg_mask = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DMG]
    is_adm = df[config.COL_ORIGINAL_VALUE] == "[ADM]"
    is_dsc = df[config.COL_ORIGINAL_VALUE] == "[DSC]"
    adm_based = is_adm.astype(float)
    dsc_based = is_dsc.astype(float)
    adm_based.loc[(~is_adm)] = None
    dsc_based.loc[(~is_dsc)] = None
    adm_based.loc[dmg_mask | is_dsc] = 0.0
    dsc_based = dsc_based.shift(
        -1, fill_value=0.0
    )  # <- "dsc_based" is shifted not to count the points of [DSC] as admitted.
    dsc_based.loc[dmg_mask | is_adm] = 0.0
    adm_based = adm_based.ffill().fillna(0.0)
    dsc_based = dsc_based.bfill().fillna(0.0)
    if ignore_truncated:
        binary_admission_status = (adm_based * dsc_based) >= 1.0
    else:
        binary_admission_status = (adm_based + dsc_based) >= 1.0
    binary_admission_status = binary_admission_status.astype(int)
    return binary_admission_status


def text_to_timedelta(col: pd.Series) -> pd.Series:
    """Converts a series of text expressions of timedelta into timedelta objects.
    This is the reverse of 'timedelta_to_text'
    """
    df = col.str.extract(
        r"(?P<years>\d+)Y/(?P<months>\d+)M/(?P<days>\d+)D (?P<hours>\d+):(?P<minutes>\d+)"
    )
    df = df.astype(int)
    days = df[config.COL_YEARS] * 365 + df[config.COL_MONTHS] * 30 + df[config.COL_DAYS]
    minutes = df[config.COL_HOURS] * 60 + df[config.COL_MINUTES]
    days = pd.to_timedelta(days, unit="D")
    minutes = pd.to_timedelta(minutes, unit="m")
    td = days + minutes
    return td


def timedelta_to_text(td_series: pd.Series) -> pd.Series:
    """Converts a series of timedelta into readable texts
    This is the reverse of 'text_to_timedelta'
    """
    total_days = td_series.dt.days
    total_minutes = td_series.dt.seconds // 60
    timedelta_cols = [
        config.COL_YEARS,
        config.COL_MONTHS,
        config.COL_DAYS,
        config.COL_HOURS,
        config.COL_MINUTES,
    ]
    td_df = pd.DataFrame(columns=timedelta_cols)
    td_df[config.COL_YEARS] = total_days // 365
    td_df[config.COL_MONTHS] = total_days % 365 // 30
    td_df[config.COL_DAYS] = total_days % 365 % 30
    td_df[config.COL_HOURS] = total_minutes // 60
    td_df[config.COL_MINUTES] = total_minutes % 60
    td_df[timedelta_cols] = td_df[timedelta_cols].astype(int).astype(str)
    for col in [config.COL_HOURS, config.COL_MINUTES]:
        td_df[col] = td_df[col].str.rjust(2, "0")
    timedelta_texts = (
        td_df[config.COL_YEARS]
        + "Y/"
        + td_df[config.COL_MONTHS]
        + "M/"
        + td_df[config.COL_DAYS]
        + "D"
        + " "
        + td_df[config.COL_HOURS]
        + ":"
        + td_df[config.COL_MINUTES]
    )
    return timedelta_texts


def extract_numeric_from_text(col: pd.Series) -> pd.Series:
    """Extracts leading numeric values from text.

    Originally designed for lab test results such as:
        "145.2 mEq/L" -> 145.2

    Returns a Series of floats. Non-numeric rows become NaN.
    """
    num_regex = r"^([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    extracted = col.str.extract(num_regex, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def get_mig_devices() -> list[str]:
    """Collects all the MIG device UUIDs.
    Returns:
        devices (list[str]): List of UUIDs.
            If there is no MIG instances, an empty list is returned.
    """
    so = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
    devices = re.findall(r"\(UUID: (MIG\-[\-0-9a-z]+)\)", so)
    return devices


def get_gpu_devices() -> list[str]:
    """Collects all the MIG device UUIDs.
    Returns:
        devices (list[str]): List of UUIDs.
            If there is no MIG instances, an empty list is returned.
    """
    so = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
    devices = re.findall(r"\(UUID: (GPU\-[\-0-9a-z]+)\)", so)
    return devices


# *********************************************
# * Functions related to reading major tables *
# *********************************************
# TODO (Yu Akagi): Because many tables are now saved as picked files, consider deprecate these functions.
def read_labelled_table(file: str) -> pd.DataFrame:
    """Reads a labelled table file."""
    # str: deidentified patient ID,unique record ID, original value
    # ts: timestamp
    # td: time available, timedelta,
    # int:row number, type, train period,test-validation period
    # float:years,months,days,hours,minutes,numeric,c0,c1,c2,c3,c4,c5,admitted,label,included

    # Read all colums as strings first
    df = pd.read_csv(file, header=0, dtype=str)

    # Integer values
    int_cols = [
        config.COL_ROW_NO,
        config.COL_TYPE,
        config.COL_TEST_PERIOD,
        config.COL_TRAIN_PERIOD,
        config.COL_INCLUDED,
    ]
    df[int_cols] = df[int_cols].astype(float).astype(int)

    # Float values
    # NOTE: Categorical value indexes (categorical_cols), labelles, and admission status are intrinsically integers; however,
    #   they need to be float values for converting to torch.Tensors. Therefore, they are converted to float here.
    categorical_cols = get_categorical_cols(df)
    float_cols = (
        config.TIMEDELTA_COMPONENT_COLS
        + [config.COL_NUMERIC]
        + categorical_cols
        + [config.COL_ADM, config.COL_LABEL]
    )
    df[float_cols] = df[float_cols].astype(float)

    # Timestamps
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])

    # Timedelta objects
    df[config.COL_TIMEDELTA] = pd.to_timedelta(df[config.COL_TIMEDELTA])
    df[config.COL_TIME_AVAILABLE] = pd.to_timedelta(df[config.COL_TIME_AVAILABLE])

    return df


def get_matrix_cols(df: pd.DataFrame) -> list[str]:
    """Gets a list of column names in a labelled table that are involved in matrix creation."""
    categorical_cols = get_categorical_cols(df)
    matrix_cols = (
        config.TIMEDELTA_COMPONENT_COLS
        + [config.COL_NUMERIC]
        + categorical_cols
        + [config.COL_ADM, config.COL_LABEL]
    )
    return matrix_cols


def read_eval_table(file: str, age_to_timedelta: bool = True) -> pd.DataFrame:
    """Reads a table for task evaluation."""
    df = pd.read_csv(
        file,
        header=0,
        dtype={
            config.COL_PID: str,
            config.COL_TYPE: int,
            config.COL_AGE: str,
            config.COL_CODE: str,
            config.COL_TEXT: str,
            config.COL_RESULT: str,
        },
    )
    period_cols = [config.COL_TRAIN_PERIOD, config.COL_TEST_PERIOD]
    for col in period_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).astype(int)
    if age_to_timedelta:
        df[config.COL_AGE] = pd.to_timedelta(df[config.COL_AGE], errors="coerce")

    return df


# *************************
# * Multiprocessing utils *
# *************************
def watch_children(processes: list[Process], watch_every: int = 1) -> bool:
    """Keeps watching child processes and returns a boolean value.
    If all the processes finishes normally, it returns true.
    It returns false otherwise.
    Args:
        processes (list[Process]): List of child processes to watch.
        watch_every (int, optional): Interval (in seconds) with which the function checks the exit codes of the processes.
            By default, the function checks every second.
    Returns:
        bool
    """
    n_proc = len(processes)
    while True:
        n_completed = 0
        for p in processes:
            if (p.exitcode is not None) and (p.exitcode != 0):
                return False
            if p.exitcode == 0:
                n_completed += 1
        if n_completed == n_proc:
            return True
        time.sleep(watch_every)
