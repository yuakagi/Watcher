"""Monte Carlo methods for down stream tasks"""

import os
import glob
import time
import pickle
import traceback
from datetime import timedelta
import pandas as pd
import torch
from ....models import WatcherOrchestrator
from ....utils import (
    LogRedirector,
    get_matrix_cols,
    get_mig_devices,
    get_gpu_devices,
)
from ....general_params import BaseSettingsManager, get_settings
from ....general_params import watcher_config as config

LOGDIR = "logs"
TASK_PROD_LOGDIR = "task_producer_logs"
TENSOR_PROD_LOGDIR = "tensor_producer_logs"
POSTPROC_LOGDIR = "postprocess_logs"


def _produce_tasks_mc_all_adm(
    files: list[tuple[str, str]],
    cleaned_adm_records: pd.DataFrame,
    output_dir: str,
    max_days: int,
    time_of_eval_td: timedelta,
    horizon_td: timedelta,
    n_iter: int,
):
    """Produces tasks for monte carlo simulations."""
    # Log config
    log_dir = os.path.join(output_dir, LOGDIR, TASK_PROD_LOGDIR)
    pid = os.getpid()
    file_name = f"log_task_producer_{pid}.txt"
    with LogRedirector(log_dir=log_dir, file_name=file_name):
        # Debug config
        debug = os.environ.get("DEBUG_MODE") == "1"
        if debug:
            files = [files[0]]
            debug_n_patients = int(os.environ.get("DEBUG_CHUNKS", 10))
            print("This is running on a debug mode")
            print(f"The number of test patients is {debug_n_patients}")
        # Initialize patient number
        patient_no = 0
        # Initialize the patient path list
        patient_list_path = os.path.join(output_dir, "patient_list.pkl")
        if os.path.exists(patient_list_path):
            # Load the existing file if the process has been restarted
            patient_path_table = pd.read_pickle(patient_list_path)
        else:
            patient_path_table = pd.DataFrame(columns=["path"])
        for labelled_file, eval_file in files:
            # Load labelled table file
            labelled_df = pd.read_pickle(labelled_file)
            matrix_cols = get_matrix_cols(labelled_df)
            # Add departments on admission
            labelled_df = pd.merge(
                labelled_df, cleaned_adm_records, on=config.COL_RECORD_ID, how="left"
            )
            # Load table for evaluation
            eval_df = pd.read_pickle(eval_file)

            # Pick up patients with admission records
            all_admissions = labelled_df.loc[
                (labelled_df[config.COL_ORIGINAL_VALUE] == "[ADM]")
                & labelled_df[config.COL_TEST_PERIOD]
                == 1
            ]
            patients = all_admissions[config.COL_PID].unique().tolist()
            # Select patients for debug
            if debug:
                patients = patients[:debug_n_patients]
            # Iterate through patients
            for p in patients:
                # Select admissions that are complete (both admitted in the period, and discharged in the period)
                table = labelled_df.loc[labelled_df[config.COL_PID] == p]
                valid_period = table[config.COL_TEST_PERIOD] == 1
                admissions = table.loc[
                    (table[config.COL_ORIGINAL_VALUE] == "[ADM]") & valid_period
                ].reset_index(drop=True)
                discharges = table.loc[
                    (table[config.COL_ORIGINAL_VALUE] == "[DSC]") & valid_period
                ].reset_index(drop=True)
                if discharges.size and admissions.size:
                    # Create rows of complete admissions
                    adm_and_dsc = pd.concat([admissions, discharges]).reset_index(
                        drop=True
                    )
                    adm_and_dsc = adm_and_dsc.sort_values(
                        by=config.COL_TIMEDELTA, ascending=True
                    )
                    adm_and_dsc["next event"] = adm_and_dsc[
                        config.COL_ORIGINAL_VALUE
                    ].shift(-1)
                    adm_and_dsc["time discharged"] = adm_and_dsc[
                        config.COL_TIMEDELTA
                    ].shift(-1)
                    valid_rows = (adm_and_dsc[config.COL_ORIGINAL_VALUE] == "[ADM]") & (
                        adm_and_dsc["next event"] == "[DSC]"
                    )
                    if valid_rows.any():
                        # Prepare directories
                        patient_no += 1
                        patient_no_str = str(patient_no).zfill(7)
                        patient_rel_path = os.path.join(
                            patient_no_str[:3], patient_no_str[3:5], patient_no_str
                        )
                        patient_dir = os.path.join(output_dir, patient_rel_path)
                        base_table_path = os.path.join(patient_dir, "full_timeline.pkl")
                        os.makedirs(patient_dir, exist_ok=True)
                        # Update the list of patient paths. This table is indexed by patient number (patient_no_str)
                        if patient_no_str not in patient_path_table.index:
                            patient_path_table.loc[patient_no_str] = [patient_rel_path]
                            patient_path_table = patient_path_table.sort_index(
                                ascending=True
                            )
                            patient_path_table.to_pickle(patient_list_path)
                        # Save a slice of evaluation table for the patient
                        sliced_eval_table = eval_df.loc[eval_df[config.COL_PID] == p]
                        sliced_eval_table.to_pickle(base_table_path)

                        # Pick up time of admissions and discharges
                        adm_and_dsc = adm_and_dsc.loc[valid_rows]
                        adm_and_dsc = adm_and_dsc.rename(
                            columns={config.COL_TIMEDELTA: "time admitted"}
                        )
                        adm_and_dsc["date admitted"] = adm_and_dsc[
                            "time admitted"
                        ].dt.floor("D")
                        adm_and_dsc["date discharged"] = adm_and_dsc[
                            "time discharged"
                        ].dt.floor("D")
                        # Select columns and reset index
                        adm_and_dsc = adm_and_dsc[
                            [
                                "time admitted",
                                "time discharged",
                                "date admitted",
                                "date discharged",
                                config.COL_TIMESTAMP,
                                config.COL_DEPT,
                            ]
                        ]
                        adm_and_dsc = adm_and_dsc.reset_index(drop=True)

                        # Iterate through admissions
                        for adm_no, row in adm_and_dsc.iterrows():
                            (
                                time_adm,
                                time_dsc,
                                date_adm,
                                date_dsc,
                                timestamp_adm,
                                dept_adm,
                            ) = row
                            adm_dir = os.path.join(patient_dir, f"admission{adm_no}")
                            n_days = (date_dsc - date_adm).days

                            # Save the general information about the admission
                            os.makedirs(adm_dir, exist_ok=True)
                            with open(
                                os.path.join(adm_dir, "admission_info.pkl"),
                                "wb",
                            ) as f:
                                info = [time_adm, time_dsc, timestamp_adm, dept_adm]
                                pickle.dump(info, f)

                            # Create input tensor, its catalog indexes, and its paths for saving
                            for day in range(0, min(n_days + 1, max_days + 1)):
                                eval_time = (
                                    date_adm + timedelta(days=day) + time_of_eval_td
                                )
                                product_file_path = os.path.join(
                                    adm_dir, f"day{day}.pkl"
                                )
                                # Skip if the file has already been created. (i.e, the process has been restarted because of an error.)
                                if not os.path.exists(product_file_path):
                                    if (time_adm <= eval_time) & (
                                        eval_time <= time_dsc
                                    ):
                                        input_mask = (
                                            table[config.COL_TIMEDELTA] <= eval_time
                                        ) & (
                                            table[config.COL_TIME_AVAILABLE]
                                            <= eval_time
                                        )
                                        matrix = table.loc[
                                            input_mask, matrix_cols
                                        ].values
                                        matrix = torch.tensor(matrix)
                                        timeline = matrix[:, :-1].unsqueeze(0).float()
                                        catalog_indexes = matrix[:, -1].tolist()
                                        product_file_path = os.path.join(
                                            adm_dir, f"day{day}.pkl"
                                        )
                                        yield timeline, catalog_indexes, n_iter, eval_time, horizon_td, product_file_path


def monte_carlo_to_all_admissions(
    dataset_dir: str,
    output_dir: str,
    blueprint: str,
    max_gpus: int = None,
    n_iter: int = 256,
    time_of_eval: int = 12,
    time_horizon: int = 7,
    max_days: int = 30,
    max_batch_size: int = 1024,
    stride: int = 64,
    max_length: int = 10000,
    compile_model: bool = False,
    restart_limit: int = 20,
):
    """Performs Monte Carlo simulations to all the admissions in the given data."""
    # Metadata
    metadata = locals().copy()

    # Log redirect
    main_log_dir = os.path.join(output_dir, LOGDIR)
    with LogRedirector(
        log_dir=main_log_dir, file_name="log_main.txt", create_dirs=True
    ):
        # Settings
        time_started = time.time()
        # NOTE: This function does not have an argument for debug mode. For debugging, set the environ var 'DEBUG_MODE' o '1' outside this function.
        settings_manager = BaseSettingsManager(
            dataset_dir=dataset_dir,
            debug=os.environ.get("DEBUG_MODE") == "1",
            debug_chunks=int(os.environ.get("DEBUG_CHUNKS", "10")),
        )
        settings_manager.write()
        try:
            # Prepare pairs of labelled and evaluation tables
            eval_tables_dir = get_settings("EVAL_TABLES_DIR")
            labelled_tables_dir = get_settings("LABELLED_TABLES_DIR")
            eval_file_pattern = os.path.join(
                eval_tables_dir, config.EVAL_TABLE_FILE_PATTERN.replace("*", "test_*")
            )
            labelled_file_pattern = os.path.join(
                labelled_tables_dir, config.LABELLED_FILE_PATTERN.replace("*", "test_*")
            )
            eval_files = glob.glob(eval_file_pattern)
            labelled_files = glob.glob(labelled_file_pattern)
            files = []
            for lf in labelled_files:
                file_parts = lf.split("_")
                file_no = file_parts[-1].replace(".pkl", "")
                paired_eval_file = os.path.join(
                    eval_tables_dir,
                    config.EVAL_TABLE_FILE_PATTERN.replace("*", f"test_{file_no}"),
                )
                if paired_eval_file in eval_files:
                    files.append((lf, paired_eval_file))
                else:
                    print(f"CAUTION: file '{paired_eval_file}' not found.")

            # Prepare cleaned admission records (load all admission records)
            cleaned_tables_dir = get_settings("CLEANED_TABLES_DIR")
            cleaned_adm_file_pattern = os.path.join(
                cleaned_tables_dir, config.ADMISSION_TABLE_PATTERN
            )
            cleaned_adm_files = glob.glob(cleaned_adm_file_pattern)
            cleaned_adm_records = pd.DataFrame(
                columns=[config.COL_RECORD_ID, config.COL_DEPT]
            )
            for af in cleaned_adm_files:
                partial_adm_df = pd.read_pickle(af)
                cleaned_adm_records = pd.concat(
                    [
                        cleaned_adm_records,
                        partial_adm_df[[config.COL_RECORD_ID, config.COL_DEPT]],
                    ]
                )
            # Prepare timedelta objects
            time_of_eval_td = timedelta(hours=time_of_eval)
            horizon_td = timedelta(days=time_horizon)
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
            # Create metadata file and save
            with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)

            # ****************
            # * Multiprocess *
            # ****************
            completed = False
            n_restart = 0
            while not completed:
                # Initialize the task generator
                task_generator = _produce_tasks_mc_all_adm(
                    files=files,
                    cleaned_adm_records=cleaned_adm_records,
                    output_dir=output_dir,
                    max_days=max_days,
                    horizon_td=horizon_td,
                    time_of_eval_td=time_of_eval_td,
                    n_iter=n_iter,
                )

                # Initialize the orchestrator
                orch = WatcherOrchestrator(
                    task_generator=task_generator,
                    log_dir=main_log_dir,
                    blueprint=blueprint,
                    gpu_ids=device_ids,
                    stride=stride,
                    max_batch_size=max_batch_size,
                    stop_vocab=None,
                    max_length=max_length,
                    return_generated_parts_only=True,
                    return_unfinished=False,
                    temperature=1.0,
                    compile_model=compile_model,
                )
                print(
                    "Successfully started all child processes for Monte Carlo simulations."
                )
                print("List of child processes:")
                print(" - task producers * 1")
                print(f" - tensor producers (GPU users) * {n_gpus}")
                print(" - postprocesser * 1")

                # Generation loop
                orch.start()
                try:
                    for product in orch:
                        df, product_file_path = product
                        df[config.COL_AGE] = pd.to_timedelta(
                            df[config.COL_AGE], errors="coerce"
                        )
                        df.reset_index(drop=True).to_pickle(product_file_path)
                    completed = True

                except Exception as e:
                    print("Error during data generation")
                    print(e)
                    print("******Traceback******")
                    traceback.print_exc()
                    print("******End Traceback******")
                    orch.terminate()
                    if n_restart < restart_limit:
                        n_restart += 1
                        continue
                    else:
                        raise RuntimeError("Failed too many times.") from e

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

        finally:
            settings_manager.delete()
