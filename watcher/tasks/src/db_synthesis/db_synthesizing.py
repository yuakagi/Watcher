"""A module to create a synthetic database"""

import os
import glob
import pickle
import traceback
import shutil
from tempfile import TemporaryDirectory
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import current_process
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from ....models import build_watcher, build_interpreter, WatcherInterpreter
from ....general_params import watcher_config as config
from ....general_params import BaseSettingsManager, get_settings
from ....utils import preprocess_timedelta_series, format_df


def _collect_ages_from_training(file: str, max_age: int):
    """Collect patient ages on first encounters."""
    # Read table
    usecols = [
        config.COL_TIMEDELTA,
        config.COL_LABEL,
        config.COL_TYPE,
        config.COL_ROW_NO,
        config.COL_INCLUDED,
    ]
    df = pd.read_pickle(file)
    df = [usecols]

    first_patient_rows = df[config.COL_ROW_NO] == 0
    # Select patients included for the training.
    included = df[config.COL_INCLUDED] == 1
    df = df[included & first_patient_rows]
    # Filter ages
    max_age_td = timedelta(days=365 * max_age)
    df = df[df[config.COL_TIMEDELTA] <= max_age_td]
    # Create a list
    ages = df[config.COL_TIMEDELTA].to_list()

    return ages


def collect_ages(
    dataset_dir: str,
    output_dir: str,
    max_age: int = 95,
    max_workers: int = None,
):
    """Collect patient initial ages in the train timelines.
    This function selectively picks up patients who were included in the training.
    The product is a 'bag of ages', from which ages are sampled for synthetic database generation.
    """
    # Configs
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir, max_workers=max_workers
    )
    settings_manager.write()
    max_workers = get_settings("MAX_WORKERS")

    # Create a bag of ages
    labelled_file_path_pattern = os.path.join(
        get_settings("LABELLED_TABLES_DIR"),
        config.LABELLED_FILE_PATTERN.replace("*", "train_*"),
    )
    files = glob.glob(labelled_file_path_pattern)
    saved_path = os.path.join(output_dir, "bag_of_ages.pkl")
    bag_of_ages = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_collect_ages_from_training, file=file, max_age=max_age)
            for file in files
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Collecting ages"
        ):
            ages = future.result()
            if ages:
                bag_of_ages += ages
    bag_of_ages = np.array(bag_of_ages)
    with open(saved_path, "wb") as f:
        pickle.dump(bag_of_ages, f)


def _synthesize_tensors_and_indexes(
    blueprint,
    bag_of_ages,
    gpu_id,
    target_per_gpu,
    batch_size,
    max_length,
    temperature,
    temp_dir,
    use_cache,
):
    """Helper function that synthesize a set of patient timelines."""
    # Configure
    file_name = f"temp_tensors_and_indexes_{gpu_id}_*.pkl"
    file_path_pattern = os.path.join(temp_dir, file_name)
    disable_pbar = gpu_id != 0

    # Instantiate a model
    model = build_watcher(blueprint=blueprint)
    model.to(f"cuda:{gpu_id}")
    print(f"Model loaded on cuda:{gpu_id}")

    # Loop
    file_no = 0
    n_generated = 0
    error_count = 0
    with torch.device(model.device):
        with tqdm(
            total=target_per_gpu, desc="[Progress on cuda:0]", disable=disable_pbar
        ) as pbar:
            while n_generated < target_per_gpu:
                try:
                    # Choose ages to initialize
                    ages = np.random.choice(bag_of_ages, batch_size, replace=True)
                    # Create arrays of initial ages
                    df = preprocess_timedelta_series(pd.Series(ages))
                    age_matrix = df.to_numpy()
                    age_rows = age_matrix.reshape(batch_size, 1, model.timedelta_dim)
                    age_rows = torch.tensor(age_rows)
                    # Create primers
                    primers = model.padding_row.repeat(batch_size, 1, 1)
                    primers[:, :, : model.timedelta_dim] = age_rows
                    primers = [t.unsqueeze(0) for t in primers]
                    catalog_idx_list = [[-1] for _ in range(batch_size)]

                    # Generate
                    timelines, catalog_indexes, _ = (
                        model.infer_autoregressively_with_cache(
                            timeline_list=primers,
                            catalog_idx_list=catalog_idx_list,
                            product_ids=None,
                            max_length=max_length,
                            stop_vocab=None,
                            max_period=None,
                            timedelta_anchor=None,
                            logits_filter="default",
                            temperature=temperature,
                            return_unfinished=False,
                            return_generated_parts_only=False,
                            show_pbar=False,
                            product_queue=None,
                            use_cache=use_cache,
                            pad_everything=False,
                        )
                    )

                    # Save data
                    file_path = file_path_pattern.replace("*", str(file_no))
                    products = [timelines, catalog_indexes]
                    with open(file_path, "wb") as f:
                        pickle.dump(products, f)

                    # Count up
                    file_no += 1
                    n_timelines = len(timelines)
                    pbar.update(n_timelines)
                    n_generated += n_timelines

                # TODO (Yu Akagi): This error handling is too general.
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    error_count += 1
                    if error_count < 10:
                        continue
                    else:
                        raise RuntimeError("Generation failed too many times")

    print(f"Generation finished by cuda:{gpu_id}")
    # Delete cache
    torch.cuda.empty_cache()
    # Delete model
    del model


def _create_tables(temp_file: str, interpreter: WatcherInterpreter, tables_dir: str):
    # Open file
    with open(temp_file, "rb") as f:
        tensors_and_indexes = pickle.load(f)
    # Get current patient numbers created by this child process
    n_patients = os.environ.get("N_PATIENTS")
    if n_patients is None:
        os.environ["N_PATIENTS"] = "0"
        n_patients = 0
    else:
        n_patients = int(n_patients)
    # Create a table
    timelines, catalog_indexes = tensors_and_indexes
    df = interpreter.create_table(
        timelines,
        catalog_indexes,
        readable_timedelta=False,
        patient_id_start=n_patients,
    )
    # Update number of patients
    last_patient_id = df[config.COL_PID].max()
    n_patients = last_patient_id + 1
    os.environ["N_PATIENTS"] = str(n_patients)
    # Add preffix to patient ids to ensure that no patient IDs overlapps among child processes
    process = current_process()
    pid = process.pid
    df[config.COL_PID] = f"{pid}_" + df[config.COL_PID].astype(str)

    # Check dtypes and column selection
    df = format_df(df, table_params=config.EVAL_TABLE_COLS)

    # Save
    file_no = temp_file.split("_")[-1].replace(".pkl", "")
    file_name = f"synthetic_ehr_table_{file_no}.pkl"
    file_path = os.path.join(tables_dir, file_name)
    df.to_pickle(file_path)


def synthesize_db(
    dataset_dir: str,
    path_to_bag_of_ages: str,
    output_dir: str,
    blueprint: str,
    target_n_patients: int = 10000,
    max_length: int = 30000,
    batch_size: int = 50,
    temperature: float = 1.0,
    max_gpus: int = None,
    max_workers: int = None,
    use_cache: bool = True,
    delete_tensors: bool = False,
):
    """Synthesize a database.

    Ages are sampled from the train data.
    This function returns nothing; however, it creates a set of tables in 'output_dir'.
    The set of patient ages sampled during the syhtesis (bag_of_ages.pkl) is also saved together with
    the tables so that they can be reused later.
    Args:
        dataset_dir (str): Path to a dataset from which ages are sampled.
        path_to_bag_of_ages (str): Path to a 'bag of ages' from which patient ages are sampled.
            Use 'collect_ages()' to create a bag of ages.
        output_dir (str): Directory for saving the synthesized database.
        blueprint (str): Blueprint of Watcher model.
        target_n_patients (int): Target number of patients included in the database.
        max_length (int): Maximum length of a timeline.
            Timelines not finished within the limits are discarded and not included in the database.
        max_age (int): Maximum patient age to start synthesis.
        batch_size (int): Batch size to perform generation. Synthesized data are saved every batch.
        temperature (float): Temperature of the model.
        max_gpus (int): Maximum number of GPUs involved in synthesis.
        max_workers (int): Number of workers (processes) involved in synthesis.
        use_cache (bool): If ture, KV-cache is used when appropriate. Default is false.
        delete_tensors (bool): If true, created tensors are delted.
    """
    # Write environment variables
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir, max_workers=max_workers
    )
    settings_manager.write()
    # Configs
    max_workers = get_settings("MAX_WORKERS")
    tables_dir = os.path.join(output_dir, "synthetic_tables")
    os.mkdir(tables_dir)
    if max_gpus is None:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = min(torch.cuda.device_count(), max_gpus)
    gpu_ids = range(0, n_gpus, 1)
    target_per_gpu = int(np.ceil(target_n_patients / n_gpus))

    # Load a bag of ages
    with open(path_to_bag_of_ages, "rb") as f:
        bag_of_ages = pickle.load(f)

    with TemporaryDirectory(dir=output_dir) as temp_dir:
        # Generate tensors and catalog indexes
        print("Generating tensors...")
        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            futures = [
                executor.submit(
                    _synthesize_tensors_and_indexes,
                    blueprint=blueprint,
                    bag_of_ages=bag_of_ages,
                    gpu_id=i,
                    target_per_gpu=target_per_gpu,
                    batch_size=batch_size,
                    max_length=max_length,
                    temperature=temperature,
                    temp_dir=temp_dir,
                    use_cache=use_cache,
                )
                for i in gpu_ids
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Creating tensors"
            ):
                _ = future.result()

        # Rename files
        temp_file_pattern = os.path.join(temp_dir, "temp_tensors_and_indexes_*.pkl")
        old_temp_files = glob.glob(temp_file_pattern)
        n_temp_files = len(os.listdir(temp_dir))
        new_temp_files = [
            os.path.join(temp_dir, f"tensors_and_indexes_{i}.pkl")
            for i in range(n_temp_files)
        ]
        for old, new in zip(old_temp_files, new_temp_files):
            os.rename(old, new)

        print(new_temp_files)

        # Create tables from the generated tensors
        interpreter = build_interpreter(blueprint=blueprint)
        print("Creating tables...")

        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            futures = [
                executor.submit(
                    _create_tables,
                    temp_file=temp_file,
                    interpreter=interpreter,
                    tables_dir=tables_dir,
                )
                for temp_file in new_temp_files
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Creating tables"
            ):
                _ = future.result()

        # Keep tensors
        if not delete_tensors:
            tensors_dir = os.path.join(output_dir, "tensors")
            if os.path.exists(tensors_dir):
                shutil.rmtree(tensors_dir)
            shutil.move(temp_dir, tensors_dir)
