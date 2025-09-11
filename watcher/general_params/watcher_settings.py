"""Module responsible for handle settings"""

import os
from datetime import datetime
import psutil
from . import watcher_config as config

# Constants
INT_VALS = [
    "MAX_WORKERS",
    "DEBUG_MODE",
    "DEBUG_CHUNKS",
    "MIN_TRAJECTORY_LENGTH",
    "MAX_SEQUENCE_LENGTH",
    "NUMERIC_BINS",
    "TD_SMALL_STEP",
    "TD_LARGE_STEP",
    "CATEGORICAL_DIM",
    "PATIENTS_PER_FILE",
]
FLOAT_VALS = [
    "TRAIN_SIZE",
    "VAL_SIZE",
    "TEST_SIZE",
]
LIST_VALS = [
    "DX_CODE_SEGMENTS",
    "MED_CODE_SEGMENTS",
    "LAB_CODE_SEGMENTS",
]
TS_VALS = {
    "TRAIN_PERIOD_START": "%Y/%m/%d %H:%M:%S",
    "UPDATE_PERIOD_START": "%Y/%m/%d %H:%M:%S",
    "TEST_PERIOD_START": "%Y/%m/%d %H:%M:%S",
    "TRAIN_PERIOD_END": "%Y/%m/%d %H:%M:%S",
    "UPDATE_PERIOD_END": "%Y/%m/%d %H:%M:%S",
    "TEST_PERIOD_END": "%Y/%m/%d %H:%M:%S",
}


def get_settings(name: str) -> str | int | float | list | datetime:
    """Loads environment variables.

    Args:
        name (str): Name of the variable.
    Returns:
        value (str|int|float|list|datetime):
            The environment variable that is converted to the proper data type.
    """
    value = os.environ.get(name)
    if value is not None:
        if name in INT_VALS:
            value = int(value)
        elif name in FLOAT_VALS:
            value = float(value)
        elif name in LIST_VALS:
            value = value.split("|")
        elif name in TS_VALS:
            value = datetime.strptime(value, TS_VALS[name])

    return value


def define_max_workers() -> int:
    """Determines the number of workers for multiprocessing.

    Returns:
        max_workers (int): Maximal number of workers to be involved in the multiprocessing.
    """
    max_workers = psutil.cpu_count(logical=config.LOGICAL_CPU) - config.CPU_MARGIN
    return max_workers


class BaseSettingsManager(object):
    """The base class to handle settings."""

    def __init__(
        self,
        dataset_dir: str = None,
        max_workers: int = None,
        debug: bool = False,
        debug_chunks: int = 8,
        **kwargs,
    ):
        # Define the minimal settings
        if max_workers is None:
            max_workers = define_max_workers()
        self.all_params = {
            "MAX_WORKERS": max_workers,
            "DEBUG_MODE": int(debug),
            "DEBUG_CHUNKS": debug_chunks,
        }

        # Add dataset-related parameters
        if dataset_dir is not None:
            traj_tensors_dir = os.path.join(dataset_dir, "timeline_tensors/")
            traj_tables_dir = os.path.join(dataset_dir, "timeline_tables/")
            labelled_dir = os.path.join(traj_tables_dir, "labelled")
            eval_tables_dir = os.path.join(traj_tables_dir, "evaluation_tables")
            intermediate_tables_dir = os.path.join(
                traj_tables_dir, "intermediate_tables"
            )
            reference_dir = os.path.join(dataset_dir, "references")
            code_counts_dir = os.path.join(reference_dir, "code_counts")
            split_params_dir = os.path.join(reference_dir, "data_split_params")
            lab_stats_dir = os.path.join(reference_dir, config.DIR_LAB_STATS)
            analytics_dir = os.path.join(dataset_dir, "analytics")
            agg_tables_dir = os.path.join(intermediate_tables_dir, "aggregated")
            fin_agg_tables_dir = os.path.join(
                intermediate_tables_dir, "finalized_aggregated"
            )
            catalogs_dir = os.path.join(reference_dir, config.DIR_CATALOGS)
            diagnosis_counts_dir = os.path.join(code_counts_dir, config.DX_CODE)
            med_code_counts_dir = os.path.join(code_counts_dir, config.MED_CODE)
            lab_code_counts_dir = os.path.join(code_counts_dir, config.LAB_CODE)
            # The main dictionary
            self.dataset_related_params = {
                # Main directories and paths
                "DATASET_DIR": dataset_dir,
                "ANALYTICS_DIR": analytics_dir,
                "TRAJECTORY_TENSORS_DIR": traj_tensors_dir,
                "TRAJECTORY_TABLES_DIR": traj_tables_dir,
                "REFERENCE_DIR": reference_dir,
                "SPLIT_PARAMS_DIR": split_params_dir,
                "DATASET_INFO_PTH": os.path.join(dataset_dir, "info.json"),
                # Analytics
                "PERFORMANCE_SHEET_PTN": os.path.join(
                    analytics_dir, "*_performance_sheet.json"
                ),
                # Tensors
                "TENSORS_TRAIN_DIR": os.path.join(traj_tensors_dir, config.TRAIN),
                "TENSORS_TEST_DIR": os.path.join(traj_tensors_dir, config.TEST),
                "TENSORS_VAL_DIR": os.path.join(traj_tensors_dir, config.VAL),
                "TENSORS_UPDATE_TRAIN_DIR": os.path.join(
                    traj_tensors_dir, "update_train/"
                ),
                "TENSORS_UPDATE_VAL_DIR": os.path.join(
                    traj_tensors_dir, "update_validation/"
                ),
                # Tables
                "CLEANED_TABLES_DIR": os.path.join(intermediate_tables_dir, "cleaned"),
                "LABELLED_TABLES_DIR": labelled_dir,
                "EVAL_TABLES_DIR": eval_tables_dir,
                "INTERMEDIATE_TABLES_DIR": intermediate_tables_dir,
                "SEQUENCED_TABLES_DIR": os.path.join(
                    intermediate_tables_dir, "sequenced"
                ),
                "AGGREGATED_TABLES_DIR": agg_tables_dir,
                "FIN_AGGREGATED_TABLES_DIR": fin_agg_tables_dir,
                "AGGREGATED_FILE_PTN": os.path.join(agg_tables_dir, "aggregated_*.pkl"),
                "FIN_AGGREGATED_FILE_PTN": os.path.join(
                    fin_agg_tables_dir, "finalized_aggregated_*.pkl"
                ),
                # Split params
                "PID_DIR": os.path.join(split_params_dir, "patient_id_lists"),
                "VISITING_DATES_DIR": os.path.join(split_params_dir, "visiting_dates"),
                # References
                "TOKEN_REFERENCE_PTH": os.path.join(
                    reference_dir, "token_reference.json"
                ),
                "TOKENIZATION_MAP_PTH": os.path.join(
                    reference_dir, "tokenization_map.json"
                ),
                "CODE_COUNTS_DIR": code_counts_dir,
                "DX_CODE_COUNTS_DIR": diagnosis_counts_dir,
                "MED_CODE_COUNTS_DIR": med_code_counts_dir,
                "LAB_CODE_COUNTS_DIR": lab_code_counts_dir,
                "DX_CODE_COUNTS_PTH": os.path.join(
                    diagnosis_counts_dir, "diagnosis_code_counts.csv"
                ),
                "MED_CODE_COUNTS_PTH": os.path.join(
                    med_code_counts_dir, "med_code_counts.csv"
                ),
                "LAB_CODE_COUNTS_PTH": os.path.join(
                    lab_code_counts_dir, "lab_code_counts.csv"
                ),
                "CATALOGS_DIR": catalogs_dir,
                "CATALOG_FILE_PTH": os.path.join(catalogs_dir, config.CATALOG_FILE),
                "CATALOG_INFO_PTH": os.path.join(
                    catalogs_dir, config.CATALOG_INFO_FILE
                ),
                "LAB_STATS_DIR": lab_stats_dir,
                "NUMERIC_STATS_PTN": os.path.join(
                    lab_stats_dir, config.LAB_NUM_STATS_PATTERN
                ),
                "NONNUMERIC_STATS_PTN": os.path.join(
                    lab_stats_dir, config.LAB_NONNUM_STATS_PATTERN
                ),
                "NUMERIC_PERCENTILES_PTN": os.path.join(
                    lab_stats_dir, config.LAB_PERCENTILES_PATTERN
                ),
            }
            self.all_params = {**self.all_params, **self.dataset_related_params}

        # Other attributes
        self.dirs_not_created = []

    def write(self):
        """Writes settings as environment variables"""
        for k, v in self.all_params.items():
            if v is not None:
                if isinstance(v, int) or isinstance(v, float):
                    os.environ[k] = str(v)
                elif isinstance(v, list):
                    os.environ[k] = "|".join(v)
                else:
                    os.environ[k] = v

    def delete(self):
        """Deletes environment variables set by this instance."""
        for k, v in self.all_params.items():
            if v is not None and k in os.environ:
                # KEEP debug-related envs
                if not k.startswith("DEBUG"):
                    del os.environ[k]

    def create_dirs(self):
        """Creates directories necessary for extraction"""
        for k, v in self.all_params.items():
            if (k.endswith("_DIR")) and (k not in self.dirs_not_created):
                if not os.path.exists(v):
                    os.makedirs(v, exist_ok=True)


class PreprocessSettingsManager(BaseSettingsManager):
    """The class to handle settings for the preprocessing steps."""

    def __init__(
        self,
        db_schema: str,
        dataset_dir: str,
        intermediate_dir: str,
        train_size: float,
        val_size: float,
        train_period: str,
        test_period: str,
        patients_per_file: int,
        update_period: str = None,
        dx_code_segments: str = None,
        med_code_segments: str = None,
        lab_code_segments: str = None,
        min_timeline_length: int = 3,
        max_sequence_length: int = 1024,
        n_numeric_bins: int = 500,
        td_small_step: int = 10,
        td_large_step: int = 120,
        max_workers: int = None,
        debug: bool = False,
        debug_chunks: int = 8,
        **kwargs,
    ):
        # Preprocess parameters
        train_period_start, train_period_end = train_period.split("-")
        train_period_start = train_period_start.strip() + " " + " 00:00:00"
        train_period_end = train_period_end.strip() + " " + "23:59:59"
        test_period_start, test_period_end = test_period.split("-")
        test_period_start = test_period_start + " " + " 00:00:00"
        test_period_end = test_period_end + " " + "23:59:59"
        if update_period is not None:
            update_period_start, update_period_end = update_period.split("-")
            update_period_start = update_period_start + " " + " 00:00:00"
            update_period_end = update_period_end + " " + "23:59:59"
        else:
            update_period_start, update_period_end = None, None
        if dx_code_segments is not None:
            dx_segs = list(dx_code_segments.split("-"))
            dx_max_dim = len(dx_segs) + 2  # +1 for [DX], +1 for [PRV]
        else:
            dx_segs = None
            dx_max_dim = 3
        if med_code_segments is not None:
            med_segs = list(med_code_segments.split("-"))
            med_max_dim = len(med_segs) + 1
        else:
            med_segs = None
            med_max_dim = 2
        if lab_code_segments is not None:
            lab_segs = list(lab_code_segments.split("-"))
            lab_max_dim = len(lab_segs) + 1
        else:
            lab_segs = None
            lab_max_dim = 2
        categorical_dim = max(dx_max_dim, med_max_dim, lab_max_dim)
        if categorical_dim > config.MAX_CATEGORICAL_DIM:
            raise ValueError(
                f"Number of code segments cannot exceed {config.MAX_CATEGORICAL_DIM-2}."
            )

        # Initialize the common variables
        super().__init__(
            dataset_dir,
            max_workers=max_workers,
            debug=debug,
            debug_chunks=debug_chunks,
        )
        # Add preprocess-specific params
        self.data_split_params = {
            "TRAIN_SIZE": train_size,
            "VAL_SIZE": val_size,
            "TEST_SIZE": 1 - (train_size + val_size),
            "TRAIN_PERIOD_START": train_period_start,
            "UPDATE_PERIOD_START": update_period_start,
            "TEST_PERIOD_START": test_period_start,
            "TRAIN_PERIOD_END": train_period_end,
            "UPDATE_PERIOD_END": update_period_end,
            "TEST_PERIOD_END": test_period_end,
            "DX_CODE_SEGMENTS": dx_segs,
            "MED_CODE_SEGMENTS": med_segs,
            "LAB_CODE_SEGMENTS": lab_segs,
            "CATEGORICAL_DIM": categorical_dim,
        }
        # Source files are csv files
        self.data_source_related_params = {
            "DB_SCHEMA": db_schema,
            "PATIENTS_PER_FILE": patients_per_file,
        }
        self.labelling_related_params = {
            "NUMERIC_BINS": n_numeric_bins,
            "TD_SMALL_STEP": td_small_step,
            "TD_LARGE_STEP": td_large_step,
        }
        self.timeline_related_params = {
            "MIN_TRAJECTORY_LENGTH": min_timeline_length,
            "MAX_SEQUENCE_LENGTH": max_sequence_length,
        }
        # Add system-related params
        temp_dir = os.path.join(intermediate_dir, "temp")
        self.temp_file_params = {
            # System-related settings
            "INTERMEDIATE_DIR": intermediate_dir,
            "TEMP_DIR": temp_dir,
            # Temporary files
            "TEMP_PKL_PTN": os.path.join(temp_dir, "temp_*.pkl"),
            "TEMP_NUM_STATS_PTN": os.path.join(temp_dir, "numeric_stats_*.pkl"),
            "TEMP_NONNUM_STATS_PTN": os.path.join(temp_dir, "nonnumeric_stats_*.pkl"),
            "TEMP_AGGREGATED_FILE_PTN": os.path.join(temp_dir, "aggregated_*.pkl"),
        }
        # Aggregate the dictionaries
        self.all_params = {
            **self.all_params,
            **self.data_split_params,
            **self.data_source_related_params,
            **self.labelling_related_params,
            **self.timeline_related_params,
            **self.temp_file_params,
        }

        # Define directories that should not be created by this class
        self.dirs_not_created += [intermediate_dir]


class TrainSettingsManager(BaseSettingsManager):
    """Class to handle settings for the training."""

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        initiated_time: str,
        max_workers: int = None,
        debug: bool = False,
        debug_chunks: int = 640,
        **kwargs,
    ):
        # Initialize variables
        super().__init__(
            dataset_dir=dataset_dir,
            max_workers=max_workers,
            debug=debug,
            debug_chunks=debug_chunks,
        )
        training_dir = os.path.join(output_dir, f"watcher_training_{initiated_time}/")
        log_dir = os.path.join(training_dir, f"logs")
        training_related_params = {
            "TRAINING_DIR": training_dir,
            "TRAINING_LOG_DIR": log_dir,
            "SNAPSHOTS_DIR": os.path.join(training_dir, config.DIR_SNAPSHOTS),
            "ACTIVE_TENSORBOARD_DIR": os.path.join(
                training_dir, config.DIR_TENSORBOARD_ACTIVE
            ),
            "MAIN_TRAINING_REPORT_PTH": os.path.join(
                training_dir, config.MAIN_TRAINING_REPORT
            ),
            "PROFILING_DIR": os.path.join(training_dir, "profiling"),
        }
        self.all_params = {**self.all_params, **training_related_params}

        # Define directories not creatred
        dirs = [d for d in self.dataset_related_params if d.endswith("_DIR")]
        self.dirs_not_created += dirs
