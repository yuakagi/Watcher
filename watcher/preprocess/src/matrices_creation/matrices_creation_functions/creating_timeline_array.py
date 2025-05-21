"""Functions to prepare tensors"""

import os
import json
import shutil
import pickle
import glob
from typing import Literal
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import Manager
from ctypes import c_int
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    add_update_period_column,
    shuffle_timeline_matrix_indexes,
    load_special_token_index,
    load_categorical_dim,
    tally_stats,
)


def _create_a_timeline_matrix(
    df: pd.DataFrame,
    shared_data_num: c_int,
    bundle_temp_dir: str,
    timeline_metadata: dict,
    categorical_dim: int,
    matrix_cols: list,
    dsc_idx: int,
    eot_idx: int,
    lab_code_token_idx: int,
    dsc_death_idx: int,
    flag: Literal["train", "validation", "test", "update_train", "update_validation"],
    lock,
) -> None:
    """Helper function that creates timeline data and metadata for a single patient.

    Args:
        df (pd.DataFrame): Dataframe that contains the full timeline sequence of a patient
        shared_data_num (ctype int): Current number of patient data. This is an integer value that is shared
            by child processes.
        bundle_temp_dir (str): Temporary directory for saving the timeline bundle data
        train (bool): Boolean flag for training data.
        update (bool): Boolean flag for update data.
        timeline_metadata (dict): Dictionary to store timeline metadata
        categorical_dim (int): Maximum number of embedding indexes to represent a code
        matrix_cols (list): List of columns of which the timeline matrix consits
        dsc_idx (int): Embedding index of [DSC] token
        eot_idx (int): Embedding index of [EOT] token
        dsc_death_idx (int): Embedding index of [DSC_EXP] token
        lab_code_token_idx (int): Embedding index of the token for medication coding system ('[LAB]')
        lock: Lock object for the perallelism.
        flag (Literal["train", "validation", "test", "update"]): Data handling mode.
    """
    # Initialize variables
    max_seq_len = get_settings("MAX_SEQUENCE_LENGTH")
    timedelta_dim = len(config.TIMEDELTA_COMPONENT_COLS)
    numeric_dim = config.NUMERIC_DIM

    # ***********************************
    # * Train-val-test-related handling *
    # ***********************************
    # Check relevant indexes
    if flag in ["train", "validation"]:
        valid_period_mask = df[config.COL_TRAIN_PERIOD] == 1
    elif flag == "test":
        valid_period_mask = df[config.COL_TEST_PERIOD] == 1
    else:
        # Update
        valid_period_mask = df[config.COL_UPDATE_PERIOD] == 1
    dmg_mask = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DMG]
    valid_mask = valid_period_mask | dmg_mask
    valid_indexes = valid_mask.values.nonzero()[0]

    # Determine the effective length
    effective_timeline_length = int(valid_mask.sum())

    # ***********************
    # * Metadata collection *
    # ***********************
    # Collect timeline metadata to save
    patient_id = df[config.COL_PID].iloc[0]
    # Create a pair of timedelta series for data augmentation
    non_td_mask = df[config.COL_YEARS].isna()
    original_td_series = df[config.COL_TIMEDELTA].copy()
    paired_td_series = original_td_series.shift(1)
    original_td_series = original_td_series.mask(non_td_mask, None)
    paired_td_series = paired_td_series.mask(non_td_mask, None)
    timedelta_series_pair = pd.concat(
        [original_td_series, paired_td_series], axis=1
    ).reset_index(drop=True)
    timedelta_series_pair.columns = ["original", "shifted"]
    # Make a copy of catalog indexes before invalid rows are replaced with config.LOGITS_IGNORE_INDEX
    catalog_indexes = df[config.COL_LABEL].tolist()
    # Check death status
    died_or_not = int(dsc_death_idx in catalog_indexes)

    # *******************
    # * Matrix creation *
    # *******************
    # Replace labels outside the relevant period with the ignored label index
    df[config.COL_LABEL] = df[config.COL_LABEL].mask(
        ~valid_mask, config.LOGITS_IGNORE_INDEX
    )
    # Create a timeline matrix out from the dataframe
    timeline_np_array = df[matrix_cols].to_numpy()
    timeline_and_labels = torch.from_numpy(timeline_np_array).float()
    # Record the length of the timeline
    timeline_length = timeline_and_labels.size(0)

    # ****************************
    # * Sequence length handling *
    # ****************************
    # Create a row for padding
    input_dim = (
        timedelta_dim + numeric_dim + categorical_dim + 1
    )  # <- +1 for admission status column
    padding_row = torch.empty(input_dim + 1)  # <- +1 for label
    padding_row[0:timedelta_dim] = torch.nan  # Pad timedelta
    padding_row[timedelta_dim] = torch.nan  # Pad numerics
    padding_row[timedelta_dim + numeric_dim : -2] = 0  # Pad categoricals
    padding_row[-2] = 0  # Admission status defaults to zero
    padding_row[-1] = config.LOGITS_IGNORE_INDEX  # Pad label

    # Pad the trajectry to the max sequence length of the model
    if timeline_length < max_seq_len:
        # Create a stack of padding rows
        padded_length = max_seq_len - timeline_length
        padded_rows = padding_row.repeat((padded_length, 1))
        # Append padding rows
        timeline_and_labels = torch.cat([timeline_and_labels, padded_rows])
        slice_points = [config.DEMOGRAPHIC_ROWS]

    # Handle long timelines
    elif timeline_length > max_seq_len:
        # Find slicable points
        non_demoraphic_indexes = valid_indexes[valid_indexes >= config.DEMOGRAPHIC_ROWS]
        min_slice_point = max(
            config.DEMOGRAPHIC_ROWS,
            non_demoraphic_indexes[0] - max_seq_len + config.DEMOGRAPHIC_ROWS + 1,
        )
        max_slice_point = max(
            min_slice_point,
            non_demoraphic_indexes[-1] - max_seq_len + config.DEMOGRAPHIC_ROWS + 1,
        )
        slice_points = list(range(min_slice_point, max_slice_point + 1))

    # Sequence length == max_seq_len
    else:
        slice_points = [config.DEMOGRAPHIC_ROWS]

    # *******************************
    # * Sequence shuffling handling *
    # *******************************
    shuffled_index_candidates = shuffle_timeline_matrix_indexes(
        timeline_and_labels,
        pad_start=timeline_length,
        dsc_idx=dsc_idx,
        eot_idx=eot_idx,
        lab_code_token_idx=lab_code_token_idx,
        k=30,
    )

    # Create a set of the matrix, slicing points and shuffled indexes
    timeline_data = [
        timeline_and_labels,
        slice_points,
        shuffled_index_candidates,
        timedelta_series_pair,
        catalog_indexes,
    ]
    # Count up the data count (this value is shared by workers)
    with lock:
        data_id = shared_data_num.value
        shared_data_num.value += 1

    # Save the data
    data_id_str = str(data_id).zfill(9)
    rel_target_path = os.path.join(
        data_id_str[0:3],
        data_id_str[3:5],
        data_id_str[5:7],
        config.TRAJECTORY_BUNDLE_FILE_PATTERN.replace("*", data_id_str),
    )
    abs_target_path = os.path.join(bundle_temp_dir, rel_target_path)
    terminal_dir = os.path.dirname(abs_target_path)
    if not os.path.exists(terminal_dir):
        os.makedirs(terminal_dir, exist_ok=True)
    with open(abs_target_path, "wb") as f:
        pickle.dump(timeline_data, f)

    # Append metadata
    timeline_metadata["data_ID"].append(data_id)
    timeline_metadata["file_path"].append(rel_target_path)
    timeline_metadata["patient_ID"].append(patient_id)
    timeline_metadata["effective_length"].append(effective_timeline_length)
    timeline_metadata["death"].append(died_or_not)


def _create_timeline_matrices(
    file: str,
    shared_data_num: c_int,
    categorical_dim: int,
    bundle_temp_dir: str,
    dsc_idx: int,
    eot_idx: int,
    lab_code_token_idx: int,
    dsc_death_idx: int,
    flag: Literal["train", "validation", "test", "update_train", "update_validation"],
    lock,
) -> dict:
    """Helper function to create patient timeline data bundles and metadata files.
    Args:
        file (str): Path to a file with labelled patient records.
        shared_data_num (ctype int): Current number of patient data. This is an integer value that is shared
            by child processes.
        categorical_dim (int): Maximum number of embedding indexes to represent a code
        bundle_temp_dir (str): Temporary directory for saving the timeline bundle data.
        train (bool): Boolean flag for training data.
        matrix_cols (list): List of columns of which the timeline matrix consits
        dsc_idx (int): Embedding index of [DSC] token
        eot_idx (int): Embedding index of [EOT] token
        lab_code_token_idx (int): Embedding index of the token for medication coding system ('[LAB]')
        dsc_death_idx (int): Embedding index of [DSC_EXP] token
        flag (Literal["train", "validation", "test", "update"]): Data handling mode.
    """
    # Initialize variables
    timeline_metadata = {
        "data_ID": [],
        "file_path": [],
        "patient_ID": [],
        "effective_length": [],
        "death": [],
    }
    categorical_cols = [f"c{i}" for i in range(categorical_dim)]
    matrix_cols = (
        config.TIMEDELTA_COMPONENT_COLS
        + [config.COL_NUMERIC]
        + categorical_cols
        + [config.COL_ADM, config.COL_LABEL]
    )

    # Load dataframe
    df = pd.read_pickle(file)

    # Count total number of patients first
    n_all_patients = df[config.COL_PID].nunique()

    # Mask for update fine-tuning data
    if flag in ["update_train", "update_validation"]:
        # Select data in the update period from the training patient data
        df = add_update_period_column(df)
        update_period_mask = df[config.COL_UPDATE_PERIOD] == 1
        dmg_mask = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DMG]
        df["effective"] = (update_period_mask | dmg_mask).astype(int)
        effective_lengths = df.groupby(config.COL_PID)["effective"].transform("sum")
        df = df.drop("effective", axis=1)
        # Overwrite 'icluded' col
        df[config.COL_INCLUDED] = (
            effective_lengths >= get_settings("MIN_TRAJECTORY_LENGTH")
        ).astype(int)

    # Count patients included/excluded
    inclusion_mask = df[config.COL_INCLUDED] == 1
    df_included = df[inclusion_mask]
    n_included = df_included[config.COL_PID].nunique()
    n_excluded = n_all_patients - n_included

    # Process timelines patient by patient
    df_included.groupby(config.COL_PID).apply(
        _create_a_timeline_matrix,
        shared_data_num=shared_data_num,
        bundle_temp_dir=bundle_temp_dir,
        timeline_metadata=timeline_metadata,
        categorical_dim=categorical_dim,
        matrix_cols=matrix_cols,
        dsc_idx=dsc_idx,
        eot_idx=eot_idx,
        lab_code_token_idx=lab_code_token_idx,
        dsc_death_idx=dsc_death_idx,
        flag=flag,
        lock=lock,
    )
    # Create stats and return
    process_analytics = {
        "analytics": {
            "included_patients": int(n_included),
            "excluded_patients": int(n_excluded),
        },
        "metadata": timeline_metadata,
    }

    return process_analytics


def create_timeline_matrices():
    """Creates patient timeline matrices and their related data series, and save them.
    The main products are a set of timeline data (timeline bundles) and its metadata table (pd.DataFrame), and
    a dictionary that stores very basic statistical data.:
        timeline data (list):
            Each entity has the following elements:
                index 0 (torch.Tensor): Matrix that contains a timeline and its labels (torch.tensor)
                        Rows outside the relevant periods (train or test) are not excluded. Instead,
                        their labels are relaced with ignored indexes (config.LOGITS_IGNORE_INDEX).
                index 1 (list[int]): List of slicable indexes, which is used for random sicing of long timelines during training
                index 2 (list[list[int]]): List of randomly shuffled index sequences, which is used during training
                index 3 (pd.DataFrame): Pair of timedelta series. The first one is the original timedelta sequence, with non-timedelta rows
                        filled with none. The other one is a shifted timedelta sequence. These two sequences are used
                        for data augmentation.
                        This dataframe is NOT padded to the max sequence length.
                index 4 (list[int]): Catalog indexes. These indexes are slightly different from labels in that indexes outside the target period
                        (train, test, update) are not replaced.
            This is saved as a pickled object.
        timeline metadata (pd.DataFrame):
            Has the following columns:
                "data_ID": Data IDs that are continuous integers from zero.
                "file_path": Relative paths for the data within the dataset folder.
                config.COL_PID: List of patient IDs.
                "effective_length": List of effective trajectry lengths.
                "sampling_point": Number of data points that the timeline has.
            This is saved as a picked object.
        analytics (dict):
            Has the following sections:
                "included_patients": Number of patients included in the dataset.
                "excluded_patients": he number of patients excluded based on their effective timeline length.
                "total_sampling_points": Number of total data sampling points.
            This is saved as a json file.

    """
    # Load variables
    categorical_dim = load_categorical_dim()
    dsc_idx = load_special_token_index("[DSC]")
    dsc_death_idx = load_special_token_index("[DSC_EXP]")
    eot_idx = load_special_token_index("[EOT]")
    lab_code_token_idx = load_special_token_index("[LAB]")

    for flag in [
        "train",
        "validation",
        "test",
        "update_train",
        "update_validation",
    ]:
        print(f"Preparing tensors for {flag}")
        # Determine output directory
        labelled_file_dir = get_settings("LABELLED_TABLES_DIR")
        if flag == "train":
            output_dir = get_settings("TENSORS_TRAIN_DIR")
        elif flag == "validation":
            output_dir = get_settings("TENSORS_VAL_DIR")
        elif flag == "test":
            output_dir = get_settings("TENSORS_TEST_DIR")
        elif flag == "update_train":
            output_dir = get_settings("TENSORS_UPDATE_TRAIN_DIR")
        else:
            output_dir = get_settings("TENSORS_UPDATE_VAL_DIR")

        # Determined the source file
        if flag == "update_train":
            labelled_file_path_pattern = os.path.join(
                labelled_file_dir,
                config.LABELLED_FILE_PATTERN.replace("*", "train_*"),
            )
        elif flag == "update_validation":
            labelled_file_path_pattern = os.path.join(
                labelled_file_dir,
                config.LABELLED_FILE_PATTERN.replace("*", "validation_*"),
            )
        else:
            labelled_file_path_pattern = os.path.join(
                labelled_file_dir,
                config.LABELLED_FILE_PATTERN.replace("*", f"{flag}_*"),
            )

        # Determine paths
        temp_dir = get_settings("TEMP_DIR")
        bundle_dir = os.path.join(output_dir, config.DIR_TRAJECTORY_BUNDLES)
        bundle_temp_dir = os.path.join(temp_dir, config.DIR_TRAJECTORY_BUNDLES)

        # Find labelled data to be processed
        labelled_files = glob.glob(labelled_file_path_pattern)
        if get_settings("DEBUG_MODE"):
            labelled_files = labelled_files[0 : get_settings("DEBUG_CHUNKS")]
        if labelled_files:
            # Execute parallelism
            max_workers = get_settings("MAX_WORKERS")
            manager = Manager()
            shared_data_num = manager.Value("i", 0)  # <- Shared among workers
            lock = manager.Lock()
            stats_list = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _create_timeline_matrices,
                        file=file,
                        shared_data_num=shared_data_num,
                        categorical_dim=categorical_dim,
                        flag=flag,
                        bundle_temp_dir=bundle_temp_dir,
                        dsc_idx=dsc_idx,
                        eot_idx=eot_idx,
                        lab_code_token_idx=lab_code_token_idx,
                        dsc_death_idx=dsc_death_idx,
                        lock=lock,
                    )
                    for file in labelled_files
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing tasks",
                ):
                    stats = future.result()
                    stats_list.append(stats)

            # Tally stats
            final_stats = tally_stats(stats_list)

            # Clear existing files and move new files
            shutil.rmtree(output_dir)
            os.mkdir(output_dir)
            shutil.move(bundle_temp_dir, bundle_dir)

            # Save the metadata
            max_sequence_length = get_settings("MAX_SEQUENCE_LENGTH")
            metadata = pd.DataFrame(final_stats["metadata"])
            metadata["sampling_point"] = np.ceil(
                metadata["effective_length"] / max_sequence_length
            )
            total_n_samplings = int(metadata["sampling_point"].sum())
            metadata = metadata.sort_values("data_ID", ascending=True)
            with open(os.path.join(output_dir, config.TRAJECTORY_METADATA), "wb") as f:
                pickle.dump(metadata, f)

            # Save stats
            analytics = final_stats["analytics"]
            analytics["total_sampling_points"] = total_n_samplings
            json_path = os.path.join(
                output_dir, config.TRAJECTORY_STATS_PATTERN
            ).replace("*", flag)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analytics, f, indent=2)

            manager.shutdown()

        else:
            print("No valid files are found. This step is skipped.")


def create_timeline_matrix_single(
    df: pd.DataFrame,
    categorical_dim: int,
) -> torch.Tensor:
    """Creates a timeline matrix for a single patient."""
    categorical_cols = [f"c{i}" for i in range(categorical_dim)]
    matrix_cols = (
        config.TIMEDELTA_COMPONENT_COLS
        + [config.COL_NUMERIC]
        + categorical_cols
        + [config.COL_ADM, config.COL_LABEL]
    )
    # Create a timeline matrix out from the dataframe
    timeline_np_array = df[matrix_cols].to_numpy(dtype=np.float64)
    timeline_and_labels = torch.tensor(timeline_np_array).float()
    return timeline_and_labels
