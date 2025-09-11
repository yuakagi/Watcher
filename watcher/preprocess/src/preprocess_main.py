"""The main module for data preprocessing"""

from __future__ import annotations

import os
import json
import tempfile
from datetime import datetime
from typing import Literal
import pandas as pd
import torch
from torch import Tensor
from .cleaning import clean_data, clean_data_single
from .split import find_visiting_dates, split_patient_ids
from .lab_value_handling import prepare_for_lab_values
from .token_handling import prepare_for_tokenization
from .sequencing import sequence_data, sequence_data_single
from .aggregating import aggregate_data, aggregate_data_single
from .label_making import make_labels, make_labels_single
from .matrices_creation import (
    create_matrices_for_pretraining,
    create_matrix_for_inference,
)
from .eval_table_creation import make_eval_tables
from ...general_params import (
    PreprocessSettingsManager,
    BaseSettingsManager,
    get_settings,
)
from ...general_params import watcher_config as config
from ...utils import (
    LogRedirector,
    load_categorical_dim,
    load_patient_id_dict,
    preprocess_timedelta_series,
)


def create_dataset(
    output_dir: str,
    train_size: float,
    val_size: float,
    train_period: str,
    test_period: str,
    max_sequence_length: int,
    db_schema: str = "public",
    min_timeline_length: int | None = None,
    patients_per_file: int = 5000,
    update_period: str = None,
    dx_code_segments: str = None,
    med_code_segments: str = None,
    lab_code_segments: str = None,
    n_numeric_bins: int = 501,
    td_small_step: int = 10,
    td_large_step: int = 60,
    max_workers: int = 1,
    log_dir: str = None,
):
    """
    Create a full dataset for model training and evaluation.

    This function performs data cleaning, temporal alignment, tokenization, aggregation,
    label generation, and matrix construction to train the Watcher model.

    Preprocessing parameters are inherited by the trained model and therefore become part of its
    effective hyperparameters.

    Warnings:
        - Some parameters of :meth:`watcher.preprocess.create_dataset` determines the model's hyperparameters.
        - Especially `max_sequence_length` determines the model's maximum sequence length.
        - Please pay attention to these settings.

    Example:
        .. code-block:: python

            from watcher.preprocess import create_dataset

            create_dataset(
                output_dir="/code/pretraining_data",
                train_size=0.8,
                val_size=0.1,
                train_period="2011/01/01-2022/12/31",
                test_period="2023/01/01-2023/12/31",
                uodate_period="2022/01/01-2022/12/31",
                db_schema="public",
                max_sequence_length=2048,
                max_workers=10,
                log_dir="/path/to/log_dir"
            )


    Train-Test Split:
        The dataset is split using a temporal strategy:

            - Patients whose records fall exclusively within the `test_period` are assigned to the test set.
            - Remaining patients are sorted by their first visit date. The most recent patients are added to the test set until the desired test size fraction (`1 - train_size - val_size`) is reached.
            - Clinical records in the training or validation set but outside the `train_period` are assigned `id = -1` and excluded from model training (i.e., ignored during loss computation).

    Medical Code Tokenization:
        You can control how medical codes are tokenized using the `dx_code_segments`, `med_code_segments`,
        and `lab_code_segments` arguments.

        By default, if these are set to `None`, each unique code is treated as a single token. However, for coding systems
        with hierarchical structure (e.g., ICD-10 or ATC), splitting codes into subtokens may improve training efficiency.

        For example, setting `dx_code_segments="1-2-1-1"` tokenizes the ICD-10 code `'J156'` into:

            ``['J****', '*15**', '***6*', '[PAD]']``

        Each token masks the non-selected characters with `*`, and unused segments are padded.
        If `dx_code_segments=None`, the same code is tokenized as:

            ``['J156']``

        This segmentation affects only the tokenization/embedding step; the original codes are still assigned
        unique vocabulary entries for modeling.

    Args:
        output_dir (str): Directory where the dataset will be saved.
        train_size (float): Fraction of patients assigned to the training set.
        val_size (float): Fraction of patients assigned to the validation set.
        train_period (str): Date range for training data (e.g., "2011/01/01-2022/12/31").
        test_period (str): Date range for test data (e.g., "2023/01/01-2023/12/31").
        max_sequence_length (int): Maximum tokenized sequence length per patient.
        db_schema (str): PostgreSQL schema name to read from.
        min_timeline_length (int, optional): Minimum number of visits required to include a patient.
            Defaults to None, which includes all patients with at least one valid clinical event.
        patients_per_file (int): Number of patients per intermediate file. Defaults to 5000.
        update_period (str, optional): Date range used to create a fine-tuning dataset for adapting the model to recent medical practice (e.g., "2022/01/01-2022/12/31").
        dx_code_segments (str, optional): Segmentation pattern for diagnosis codes (e.g., "1-2-1-1").
        med_code_segments (str, optional): Segmentation pattern for medication codes (e.g., "1-2-1-1-2").
        lab_code_segments (str, optional): Segmentation pattern for laboratory test codes.
        n_numeric_bins (int): Number of bins for discretizing numeric values. Defaults to 601.
        td_small_step (int): Step size (in minutes) to discretize time progression from +0 to +1440 min (1 day).
        td_large_step (int): Step size in minutes to discretize a 24-hour day from 00:00 to 23:59 based on clock time.
        max_workers (int): Maximum number of parallel workers. Defaults to 1.
        log_dir (str, optional): Directory to store preprocessing logs.

    Returns:
        None
    """

    # Debugging config
    debug_flag = os.environ.get("DEBUG_MODE")
    debug = debug_flag == "1"
    debug_chunks = 8 if debug else 0

    # Preprocess args
    if min_timeline_length is None:
        min_timeline_length = config.DEMOGRAPHIC_ROWS + 1

    # Create a temp dir in the output directory if not specified
    with LogRedirector(log_dir=log_dir, file_name=None):
        with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
            # Variables
            dataset_dir = os.path.join(output_dir, "dataset")
            saved_setting_kwargs = {
                "dataset_dir": dataset_dir,
                "intermediate_dir": temp_dir,
                "db_schema": db_schema,
                "patients_per_file": patients_per_file,
                "train_size": train_size,
                "val_size": val_size,
                "train_period": train_period,
                "test_period": test_period,
                "update_period": update_period,
                "dx_code_segments": dx_code_segments,
                "med_code_segments": med_code_segments,
                "lab_code_segments": lab_code_segments,
                "min_timeline_length": min_timeline_length,
                "max_sequence_length": max_sequence_length,
                "n_numeric_bins": n_numeric_bins,
                "td_small_step": td_small_step,
                "td_large_step": td_large_step,
                "max_workers": max_workers,
                "debug": debug,
                "debug_chunks": debug_chunks,
            }
            unsaved_setting_kwargs = {}
            # Initialize preprocessing settings
            settings_manager = PreprocessSettingsManager(
                **saved_setting_kwargs, **unsaved_setting_kwargs
            )
            settings_manager.create_dirs()
            settings_manager.write()

            try:
                # Preprocess steps
                clean_data()
                find_visiting_dates()
                split_patient_ids()
                prepare_for_lab_values()
                prepare_for_tokenization()
                sequence_data()
                aggregate_data()
                make_labels()
                make_eval_tables()
                create_matrices_for_pretraining()

                # Write down the parameters of the dataset preparation
                info_dict = {}
                info_dict["last_updated"] = datetime.strftime(
                    datetime.now(), "%Y/%m/%d %H:%M:%S"
                )
                info_dict["dataset_params"] = saved_setting_kwargs
                # Write the part of model hyperparameters determined by the dataset
                token_ref_path = get_settings("TOKEN_REFERENCE_PTH")
                with open(token_ref_path, "r", encoding="utf-8") as f:
                    token_reference = json.load(f)
                input_vocab_size = token_reference["vocabulary_size"]
                info_dict["hyperparameters"] = {
                    "categorical_dim": load_categorical_dim(),
                    "input_vocab_size": input_vocab_size,
                    "max_sequence_length": max_sequence_length,
                    "n_numeric_bins": n_numeric_bins,
                    "td_small_step": td_small_step,
                    "td_large_step": td_large_step,
                }
                # Save the info
                info_path = get_settings("DATASET_INFO_PTH")
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(info_dict, f, ensure_ascii=False, indent=2)

            finally:
                settings_manager.delete()


def preprocess_for_inference(
    patient_id: str,
    model: Watcher,
    start: str | None = None,
    end: str | None = None,
    db_schema: str = "public",
) -> tuple[Tensor, list[int], datetime]:
    """
    Preprocess a single patient's records for inference using a Watcher model.

    This function performs per-patient data cleaning, sequencing, and label integration,
    then transforms the results into a tensor input and catalog index list for model input.

    Args:
        patient_id (str): Patient identifier.
        model (Watcher): A Watcher model instance (required for categorical configuration).
        start (str, optional): Start timestamp in "YYYY/mm/dd HH:MM" format.
        end (str, optional): End timestamp in "YYYY/mm/dd HH:MM" format.
        db_schema (str): PostgreSQL schema to query from.

    Returns:
        tuple:
            - timeline (Tensor): Input tensor of shape (1, seq_len, dim) excluding label index.
            - catalog_indexes (list[int]): Label indices for trajectory steps.
            - dob (datetime): Date of birth (used internally for label alignment).

    Examples:
        .. code-block:: python

            from watcher.preprocess import preprocess_for_inference

            timeline, catalog, dob = preprocess_for_inference(
                patient_id="123456",
                model=watcher_model,
                start="2020/01/01 00:00",
                end="2021/01/01 00:00"
            )
    """
    # TODO: Replace `model` with `catalog`.

    # Preprocess steps
    tables, dob = clean_data_single(
        patient_id=patient_id,
        start=start,
        end=end,
        db_schema=db_schema,
    )
    tables = sequence_data_single(tables=tables, model=model)
    df = aggregate_data_single(categorical_dim=model.categorical_dim, tables=tables)
    df = make_labels_single(agg_df=df, model=model)
    timeline_and_labels = create_matrix_for_inference(
        df=df, categorical_dim=model.categorical_dim
    )
    timeline = timeline_and_labels[:, :-1].unsqueeze(0)
    catalog_indexes = timeline_and_labels[:, -1].long().tolist()

    # Handling for edge cases
    if torch.isnan(timeline[0, 0, 0]):
        if end is not None:
            # If the initial age row is null (No records other than demographics),
            # Create the first age row using 'end'
            end_date = datetime.strptime(end, "%Y%m%d %H%M").date()
            end_diff = end_date - dob
            first_age_vals = (
                preprocess_timedelta_series(pd.Series([end_diff])).iloc[0].values
            )
            first_age_vals = torch.from_numpy(first_age_vals)
            timeline[0, 0, : first_age_vals.size(0)] = first_age_vals
        else:
            # Current implementation relies on 'end'. Therefore, this is skipped if end is None.
            # TODO (Yu Akagi): Make this handlings available without 'end'.
            pass

    print("END")

    return timeline, catalog_indexes, dob


def get_patient_ids(
    dataset_dir: str, group: Literal["train", "validation", "test", "all"]
) -> list[str]:
    """
    Loads patient IDs used for training, validation, testing, or all patient IDs.

    Example:
        .. code-block:: python

            from watcher.preprocess import get_patient_ids

            patient_ids = get_patient_ids(dataset_dir="/path/to/dataset", group="train")
            print(patient_ids[:10])

    Args:
        dataset_dir (str): Path to the dataset directory.
        group (Literal["train", "validation", "test", "all"]): Group to load patient IDs from.
            - "train": Training set only.
            - "validation": Validation set only.
            - "test": Test set only.
            - "all": Combine training, validation, and test sets.

    Returns:
        list[str]: List of patient IDs corresponding to the specified group.

    """
    settings_manager = BaseSettingsManager(dataset_dir=dataset_dir)
    settings_manager.write()
    try:
        included_only = group in ["train", "validation"]
        id_dict = load_patient_id_dict(included_only=included_only)
        if group == "all":
            patient_ids = id_dict["train"] + id_dict["validation"] + id_dict["test"]
        else:
            patient_ids = id_dict[group]

        return patient_ids
    finally:
        settings_manager.delete()
