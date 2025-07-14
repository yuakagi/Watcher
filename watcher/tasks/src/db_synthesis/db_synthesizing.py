"""A module to create a synthetic database"""

import os
import glob
import pickle
import random
import time
import math
import json
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from ....models import build_watcher, generate_from_batch
from ....general_params import watcher_config as config
from ....general_params import BaseSettingsManager, get_settings
from ....utils import (
    get_mig_devices,
    get_gpu_devices,
    preprocess_timedelta_series,
    LogRedirector,
)

LOGDIR = "logs"
BOA_FILE_NAME = "bag_of_ages.pkl"
SYNTHETIC_DATA_DIR = "synthetic_db"


def _collect_ages_from_training(
    file: str, min_age: int, max_age: int, included_only: bool
) -> list[timedelta]:
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
    df = df[usecols]

    # Select first rows onlu
    first_patient_rows = df[config.COL_ROW_NO] == 0
    df = df.loc[first_patient_rows]
    # Select patients included for the training.
    if included_only:
        included = df[config.COL_INCLUDED] == 1
        df = df.loc[included]
    # Filter ages
    max_age_td = timedelta(days=365 * max_age)
    min_age_td = timedelta(days=365 * min_age)
    df = df[df[config.COL_TIMEDELTA] <= max_age_td]
    df = df[df[config.COL_TIMEDELTA] >= min_age_td]
    # Create a list
    ages = df[config.COL_TIMEDELTA].dt.to_pytimedelta().tolist()

    return ages


def collect_ages(
    dataset_dir: str,
    output_dir: str,
    min_age: int = 18,
    max_age: int = 90,
    max_workers: int = 1,
    included_only: bool = False,
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
    saved_path = os.path.join(output_dir, BOA_FILE_NAME)
    bag_of_ages = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _collect_ages_from_training,
                file=file,
                min_age=min_age,
                max_age=max_age,
                included_only=included_only,
            )
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

    print("Bag-of-age file saved:", saved_path)


def _produce(
    output_dir: str,
    blueprint: str,
    gpu_id: str,
    bag_of_age: list[timedelta],
    n_targets: int,
    batch_size: int,
    stride: int,
    max_length: int,
    add_noise: bool,
):

    # Get process id
    process_id = str(os.getpid())
    # Log redirect
    main_log_dir = os.path.join(output_dir, LOGDIR)
    with LogRedirector(
        log_dir=main_log_dir, file_name=f"{process_id}_log.txt", create_dirs=True
    ):

        # Instantiate
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        model = build_watcher(blueprint=blueprint, train=False)
        model = model.to("cuda")

        # Generate
        n_generated = 0
        with tqdm(
            total=n_targets,
            desc=f"Generating data on GPU {gpu_id}",
            unit="patients",
        ) as pbar:
            while n_generated < n_targets:
                # Create initial rows
                sampled_ages = []
                for _ in range(batch_size):
                    # Sample ages
                    age = random.choice(bag_of_age)
                    if add_noise:
                        # Add noise
                        years = age.total_seconds() // (365.25 * 24 * 60 * 60)
                        noise = random.uniform(-1, 1) * years * 0.05  # ± 5% noise
                        age = max(
                            timedelta(seconds=0), age + timedelta(days=noise * 365.25)
                        )
                    # Round ages to days
                    age = timedelta(days=round(age.total_seconds() // (24 * 60 * 60)))
                    sampled_ages.append(age)
                # Create a DataFrame
                df = preprocess_timedelta_series(pd.Series(sampled_ages))
                primers = model.padding_row.repeat(batch_size, 1)
                primers[:, : model.numeric_start] = torch.from_numpy(df.values).to(
                    primers.device
                )

                primers = primers.reshape(batch_size, 1, -1).float()
                del df
                # Generate
                synthetic_patient_data = generate_from_batch(
                    model=model,
                    timeline_batch=primers,
                    catalog_ids_batch=[[-1] for _ in range(batch_size)],
                    time_horizon=None,
                    horizon_start=None,
                    stride=stride,
                    max_length=max_length,
                    return_generated_parts_only=False,
                    return_unfinished=False,
                    compile_model=False,
                    quantize_weights=False,
                    logits_filter="default",
                    temperature=1.0,
                    show_pbar=False,
                    compiled_fn=None,
                )

                # Set patient IDs
                synthetic_patient_data["patient_id"] = (
                    synthetic_patient_data["patient_id"].factorize()[0].astype(int)
                    + n_generated
                ).astype(str)
                synthetic_patient_data["patient_id"] = (
                    process_id + "_" + synthetic_patient_data["patient_id"]
                )
                # Save files
                for patient_id in synthetic_patient_data["patient_id"].unique():
                    # Slice the data for the patient
                    patient_data = synthetic_patient_data.loc[
                        synthetic_patient_data["patient_id"] == patient_id
                    ]
                    patient_data = patient_data.reset_index(drop=True)
                    # Save the data
                    file_path = os.path.join(
                        output_dir,
                        SYNTHETIC_DATA_DIR,
                        process_id,
                        str(n_generated),
                        f"synthetic_patient_{patient_id}.pkl",
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    patient_data.to_pickle(file_path)

                # Count generated patients
                n_generated += synthetic_patient_data["patient_id"].nunique()
                pbar.update(
                    synthetic_patient_data["patient_id"].nunique()
                )  # Update progress bar
                # Clear memory
                torch.cuda.empty_cache()  # Clear GPU memory
                del synthetic_patient_data

    return n_generated


def synthesize_database(
    dataset_dir: str,
    output_dir: str,
    blueprint: str,
    n_targets: int,
    max_gpus: int = None,
    add_age_noise: bool = True,
    batch_size: int = 256,
    stride: int = 64,
    max_length: int = 100000,
    min_age: int = 18,
    max_age: int = 90,
    age_collection_max_workers: int = 1,
    age_collection_included_only: bool = False,
):
    """Synthesize a synthetic database using the provided blueprint and bag of ages.

    Args:
        dataset_dir (str): Path to the dataset directory.
        output_dir (str): Path to the output directory where the synthetic database will be saved.
        blueprint (str): Path to the blueprint file.
        n_targets (int): Number of synthetic patients to generate.
        max_gpus (int, optional): Maximum number of GPUs to use. Defaults to None
        add_age_noise (bool, optional): Whether to add noise to the ages. Defaults to True.
        batch_size (int, optional): Batch size for generation. Defaults to 256.
        stride (int, optional): Stride for generation. Defaults to 64.
        max_length (int, optional): Maximum length of the generated timelines. Defaults to 100000.
        min_age (int, optional): Minimum age of the patients. Defaults to 18.
        max_age (int, optional): Maximum age of the patients. Defaults to 90.
        age_collection_max_workers (int, optional): Number of workers to use for age collection. Defaults to 1.
        age_collection_included_only (bool, optional): Whether to include only patients who were included
            in the training. Defaults to False.
    """
    # Create metadata file and save
    metadata = locals().copy()
    metadata["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    # Set timer
    time_started = time.time()
    # Log redirect
    main_log_dir = os.path.join(output_dir, LOGDIR)
    with LogRedirector(
        log_dir=main_log_dir, file_name="log_main.txt", create_dirs=True
    ):
        # Collect ages
        boa_path = os.path.join(output_dir, BOA_FILE_NAME)
        if not os.path.exists(boa_path):
            print("Collecting ages...")
            # Collect ages
            collect_ages(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                min_age=min_age,
                max_age=max_age,
                max_workers=age_collection_max_workers,
                included_only=age_collection_included_only,
            )
        else:
            print("Bag of ages already exists, skipping collection.")
        # Load bag of ages
        with open(boa_path, "rb") as f:
            bag_of_ages = pickle.load(f)

        # GPU configuration
        mig_devices = get_mig_devices()
        gpu_devices = get_gpu_devices()
        device_ids = mig_devices + gpu_devices
        if max_gpus is None:
            n_gpus = len(device_ids)
        else:
            n_gpus = min(max_gpus, len(device_ids))
        device_ids = device_ids[:n_gpus]
        print("GPUs used:")
        for did in device_ids:
            print(did)

        # ****************
        # * Multiprocess *
        # ****************
        print("List of child processes:")
        for i, gpu_id in enumerate(device_ids):
            print(f" - tensor producer (GPU user) * {i + 1} * {gpu_id}")

        n_sub_targets = math.ceil(n_targets / n_gpus)
        with ProcessPoolExecutor(max_workers=n_gpus) as executor:
            futures = []
            for i, gpu_id in enumerate(device_ids):
                # Create a future for each GPU
                futures.append(
                    executor.submit(
                        _produce,
                        output_dir=output_dir,
                        blueprint=blueprint,
                        gpu_id=gpu_id,
                        bag_of_age=bag_of_ages,
                        n_targets=n_sub_targets,
                        batch_size=batch_size,
                        stride=stride,
                        max_length=max_length,
                        add_noise=add_age_noise,
                    )
                )

            # Collect results
            n_generated = 0
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Generating data"
            ):
                sub_n_gen = future.result()
                n_generated += sub_n_gen

        print("All processes finished")
        time_end = time.time()
        time_diff = time_end - time_started
        days = time_diff // (60 * 60 * 24)
        hours = time_diff % (60 * 60 * 24) // (60 * 60)
        mins = time_diff % (60 * 60 * 24) % (60 * 60) // 60
        secs = time_diff % (60 * 60 * 24) % (60 * 60) % 60
        print(
            f"Total time spent: {days} days, {hours} hrs, {mins} mins, {secs} seconds"
        )

        # Update metadata
        metadata["total_time_spemt"] = (
            f"{days} days, {hours} hrs, {mins} mins, {secs} seconds"
        )
        metadata["n_generated"] = n_generated
        with open(
            os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)
