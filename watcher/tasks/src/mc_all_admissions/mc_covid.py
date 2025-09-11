"""Monte Carlo methods for down stream tasks"""

import os
import glob
import time
import pickle
import traceback
from datetime import timedelta
import psycopg
import pandas as pd
import numpy as np
import torch
from ....preprocess import preprocess_for_inference
from ....models import WatcherOrchestrator, build_watcher
from ....utils import (
    LogRedirector,
    load_db_params,
    get_mig_devices,
    get_gpu_devices,
)
from ....general_params import BaseSettingsManager, get_settings
from ....general_params import watcher_config as config

LOGDIR = "logs"
TASK_PROD_LOGDIR = "task_producer_logs"
TENSOR_PROD_LOGDIR = "tensor_producer_logs"
POSTPROC_LOGDIR = "postprocess_logs"

N_ITER = 256
HORIZON = 7

DB_PARAMS = dict(
    user="twinadmin",
    password="mypassword",
    host="db",
    port="5432",
    dbname="twin",
)
DB_PARAMS_INF = load_db_params()

COVID_QUERY = """
SELECT DISTINCT a.patient_id, a.timestamp AS admission_time, l.timestamp AS lab_time
FROM admissions a
JOIN laboratory_results l
  ON a.patient_id = l.patient_id
JOIN patients p
  ON a.patient_id = p.patient_id
WHERE a.timestamp >= '2023-01-01'
  AND l.item_code IN (
    '5F6251411063---11',
    '5F6251413063---01',
    '5F6251450061---11',
    '5F6251450063---11'
  )
  AND l.nonnumeric = '(+)' 
  AND ABS(EXTRACT(EPOCH FROM (l.timestamp - a.timestamp))) <= 86400
  AND AGE(a.timestamp, p.date_of_birth) >= INTERVAL '20 years';

"""

CRP_QUERY = """
WITH covid_admissions AS (
  SELECT
    a.unique_record_id,
    a.patient_id,
    a.timestamp AS admission_time,
    l.timestamp AS covid_time
  FROM admissions a
  JOIN laboratory_results l
    ON a.patient_id = l.patient_id
  JOIN patients p
    ON a.patient_id = p.patient_id
  WHERE a.timestamp >= '2023-01-01'
    AND l.item_code IN (
      '5F6251411063---11',
      '5F6251413063---01',
      '5F6251450061---11',
      '5F6251450063---11'
    )
    AND l.nonnumeric = '(+)' 
    AND l.timestamp BETWEEN a.timestamp - interval '24 hour' AND a.timestamp + interval '24 hour'
    AND AGE(a.timestamp, p.date_of_birth) >= INTERVAL '20 years'
),
first_crp_after AS (
  SELECT DISTINCT ON (ca.unique_record_id)
    ca.unique_record_id,
    ca.patient_id,
    ca.admission_time,
    ca.covid_time,
    crp.timestamp AS crp_time
  FROM covid_admissions ca
  JOIN laboratory_results crp
    ON ca.patient_id = crp.patient_id
  WHERE crp.item_code = '5C0700000023---01'
    AND crp.timestamp > ca.admission_time
    AND crp.timestamp > ca.covid_time
  ORDER BY ca.unique_record_id, crp.timestamp
)
SELECT *
FROM first_crp_after;
"""


CRE_QUERY = """
WITH covid_admissions AS (
  SELECT
    a.unique_record_id,
    a.patient_id,
    a.timestamp AS admission_time,
    l.timestamp AS covid_time
  FROM admissions a
  JOIN laboratory_results l
    ON a.patient_id = l.patient_id
  JOIN patients p
    ON a.patient_id = p.patient_id
  WHERE a.timestamp >= '2023-01-01'
    AND l.item_code IN (
      '5F6251411063---11',
      '5F6251413063---01',
      '5F6251450061---11',
      '5F6251450063---11'
    )
    AND l.nonnumeric = '(+)' 
    AND l.timestamp BETWEEN a.timestamp - interval '24 hour' AND a.timestamp + interval '24 hour'
    AND AGE(a.timestamp, p.date_of_birth) >= INTERVAL '20 years'
),
first_cre_after AS (
  SELECT DISTINCT ON (ca.unique_record_id)
    ca.unique_record_id,
    ca.patient_id,
    ca.admission_time,
    ca.covid_time,
    cre.timestamp AS cre_time
  FROM covid_admissions ca
  JOIN laboratory_results cre
    ON ca.patient_id = cre.patient_id
  WHERE cre.item_code = '3C0150000023---01'
    AND cre.timestamp > ca.admission_time
    AND cre.timestamp > ca.covid_time
  ORDER BY ca.unique_record_id, cre.timestamp
)
SELECT *
FROM first_cre_after;
"""


def _produce_tasks_mc_all_adm(
    blueprint: str,
    output_dir: str,
):
    """Produces tasks for monte carlo simulations."""
    # Vars
    dummy_model = build_watcher(blueprint, train=False).cpu()
    catalog = dummy_model.interpreter.catalogs["full"]
    crp_id = catalog.loc[
        catalog[config.COL_ORIGINAL_VALUE] == "5C0700000023---01", config.COL_LABEL
    ].item()
    cre_id = catalog.loc[
        catalog[config.COL_ORIGINAL_VALUE] == "3C0150000023---01", config.COL_LABEL
    ].item()
    percentiles = dummy_model.interpreter.percentiles
    percentile_steps = dummy_model.interpreter.percentile_steps
    percentile_cols = dummy_model.interpreter.percentile_cols
    crp_percentiles = (
        percentiles.loc[
            percentiles[config.COL_ITEM_CODE] == "5C0700000023---01", percentile_cols
        ]
        .values.astype(float)
        .reshape(-1)
    )
    cre_percentiles = (
        percentiles.loc[
            percentiles[config.COL_ITEM_CODE] == "3C0150000023---01", percentile_cols
        ]
        .values.astype(float)
        .reshape(-1)
    )
    min_td_idx = dummy_model.interpreter.min_indexes["timedelta"]
    min_num_idx = dummy_model.interpreter.min_indexes["numeric_lab_values"]

    # Log config
    log_dir = os.path.join(output_dir, LOGDIR, TASK_PROD_LOGDIR)
    pid = os.getpid()
    file_name = f"log_task_producer_{pid}.txt"

    with LogRedirector(log_dir=log_dir, file_name=file_name):

        # Query all admissions first
        with psycopg.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute(COVID_QUERY)
                query_results = cur.fetchall()
                print("Number of CRP-after-COVID admissions:", len(query_results))
                # Define column names to match your SELECT fields
                columns = [
                    "patient_id",
                    "admission_time",
                    "covid_time",
                ]
                # Convert to DataFrame
                covid_adms = pd.DataFrame(query_results, columns=columns)

                cur.execute(CRP_QUERY)
                query_results = cur.fetchall()
                print("Number of CRP-after-COVID admissions:", len(query_results))
                # Define column names to match your SELECT fields
                columns = [
                    "admission_id",
                    "patient_id",
                    "admission_time",
                    "covid_time",
                    "crp_time",
                ]
                # Convert to DataFrame
                crp_adms = pd.DataFrame(query_results, columns=columns)

                cur.execute(CRE_QUERY)
                query_results = cur.fetchall()
                print("Number of CRE-after-COVID admissions:", len(query_results))
                # Define column names to match your SELECT fields
                columns = [
                    "admission_id",
                    "patient_id",
                    "admission_time",
                    "covid_time",
                    "cre_time",
                ]
                # Convert to DataFrame
                cre_adms = pd.DataFrame(query_results, columns=columns)

        # Debug config
        debug = os.environ.get("DEBUG_MODE") == "1"
        if debug:
            covid_adms = covid_adms[:4]
            crp_adms = crp_adms[:4]
            cre_adms = cre_adms[:4]

        # ******************** COVID ALL ****************
        cov_dir = os.path.join(output_dir, "all_covid")
        os.makedirs(cov_dir, exist_ok=True)
        for row_no, row in covid_adms.iterrows():
            patient_id, adm_time, covid_time = row
            # Create dir
            patient_dir = os.path.join(cov_dir, patient_id)
            adm_dir = os.path.join(patient_dir, adm_time.strftime("%Y%m%d"))
            os.makedirs(adm_dir, exist_ok=True)
            inf_start = max(adm_time, covid_time)
            inf_start_str = inf_start.strftime(config.DATETIME_FORMAT)
            timeline, catalog_indexes, dob = preprocess_for_inference(
                patient_id=patient_id,
                model=dummy_model,
                start=None,
                end=inf_start_str,
            )
            horizon_start_base = inf_start - dob

            # Sex
            sex_id = timeline[0, 1, 6].long().item()
            sex = 1 if sex_id == 11 else 0  # 1 if female 0 if male

            # Adding ages
            for age_delta in [0, 5, 10, 15]:
                modified_tl = timeline.clone()
                scaled_age_delta = age_delta / 120
                age_mask = ~torch.isnan(modified_tl[:, :, 0])
                modified_tl[:, :, 0][age_mask] += scaled_age_delta
                horizon_start = horizon_start_base + timedelta(days=365 * age_delta)
                modified_age = int(modified_tl[:, :, 0][age_mask][-1].item() * 120)
                product_file_path = os.path.join(
                    adm_dir,
                    f"sim_age_added_{age_delta}_age{modified_age}_sex{sex}_{N_ITER}iter_horizon{HORIZON}.pkl",
                )

                yield modified_tl, catalog_indexes, N_ITER, horizon_start, timedelta(
                    days=HORIZON
                ), product_file_path

        # *************** CRP counterfactuals ****************
        crp_dir = os.path.join(output_dir, "crp")
        os.makedirs(crp_dir, exist_ok=True)
        for row_no, row in crp_adms.iterrows():
            adm_id, patient_id, adm_time, covid_time, crp_time = row
            crp_time_str = crp_time.strftime(config.DATETIME_FORMAT)
            # Create dir
            patient_dir = os.path.join(crp_dir, patient_id)
            adm_dir = os.path.join(patient_dir, adm_time.strftime("%Y%m%d"))
            os.makedirs(adm_dir, exist_ok=True)
            timeline, catalog_indexes, dob = preprocess_for_inference(
                patient_id=patient_id,
                model=dummy_model,
                start=None,
                end=crp_time_str,
            )
            horizon_start = crp_time - dob

            # Sex
            sex_id = timeline[0, 1, 6].long().item()
            sex = 1 if sex_id == 11 else 0  # 1 if female 0 if male

            # Remove lab items other than CRP
            crp_pos = int((np.array(catalog_indexes) == crp_id).nonzero()[0][-1])
            last_td_pos = int(
                (np.array(catalog_indexes) >= min_td_idx).nonzero()[0][-1]
            )
            scaled_crp_val = timeline[0, crp_pos + 1, 5].item()
            unsclaed_crp_val_idx = np.argmin(np.abs(percentile_steps - scaled_crp_val))
            unsclaed_crp_val = crp_percentiles[unsclaed_crp_val_idx]

            new_timeline = torch.cat(
                [
                    timeline[:, : last_td_pos + 1, :],
                    timeline[:, crp_pos : crp_pos + 2, :],
                ],
                dim=1,
            )
            new_catalog_indexes = (
                catalog_indexes[: last_td_pos + 1]
                + catalog_indexes[crp_pos : crp_pos + 2]
            )

            # Adding crp
            for crp_delta in [0, 5, 10, 15, 20]:
                # Copy
                modified_tl = new_timeline.clone()
                modified_ci = new_catalog_indexes.copy()
                # Modify crp vals
                modified_crp = unsclaed_crp_val + crp_delta
                modified_crp_val_idx = np.argmin(np.abs(crp_percentiles - modified_crp))
                modified_cat_id = min_num_idx + modified_crp_val_idx
                sclaed_modified_crp = percentile_steps[modified_crp_val_idx]
                # Replace timeline & catalog ids
                modified_tl[:, -1, 5] = sclaed_modified_crp
                modified_ci[-1] = modified_cat_id

                # File name
                product_file_path = os.path.join(
                    adm_dir,
                    f"sim_CRP_added_{crp_delta}_crp{modified_crp}_sex{sex}_{N_ITER}iter_horizon{HORIZON}.pkl",
                )

                if not os.path.exists(product_file_path):
                    yield modified_tl, modified_ci, N_ITER, horizon_start, timedelta(
                        days=HORIZON
                    ), product_file_path

        # *************** CRE counterfactuals ****************
        cre_dir = os.path.join(output_dir, "cre")
        os.makedirs(cre_dir, exist_ok=True)
        for row_no, row in cre_adms.iterrows():
            adm_id, patient_id, adm_time, covid_time, cre_time = row
            cre_time_str = cre_time.strftime(config.DATETIME_FORMAT)
            # Create dir
            patient_dir = os.path.join(cre_dir, patient_id)
            adm_dir = os.path.join(patient_dir, adm_time.strftime("%Y%m%d"))
            os.makedirs(adm_dir, exist_ok=True)
            timeline, catalog_indexes, dob = preprocess_for_inference(
                patient_id=patient_id,
                model=dummy_model,
                start=None,
                end=cre_time_str,
            )
            horizon_start = cre_time - dob

            # Sex
            sex_id = timeline[0, 1, 6].long().item()
            sex = 1 if sex_id == 11 else 0  # 1 if female 0 if male

            # Remove lab items other than cre
            cre_pos = int((np.array(catalog_indexes) == cre_id).nonzero()[0][-1])
            last_td_pos = int(
                (np.array(catalog_indexes) >= min_td_idx).nonzero()[0][-1]
            )
            scaled_cre_val = timeline[0, cre_pos + 1, 5].item()
            unsclaed_cre_val_idx = np.argmin(np.abs(percentile_steps - scaled_cre_val))
            unsclaed_cre_val = cre_percentiles[unsclaed_cre_val_idx]

            new_timeline = torch.cat(
                [
                    timeline[:, : last_td_pos + 1, :],
                    timeline[:, cre_pos : cre_pos + 2, :],
                ],
                dim=1,
            )
            new_catalog_indexes = (
                catalog_indexes[: last_td_pos + 1]
                + catalog_indexes[cre_pos : cre_pos + 2]
            )

            # Adding cre
            for cre_delta in [0, 1, 3, 5]:
                # Copy
                modified_tl = new_timeline.clone()
                modified_ci = new_catalog_indexes.copy()
                # Modify cre vals
                modified_cre = unsclaed_cre_val + cre_delta
                modified_cre_val_idx = np.argmin(np.abs(cre_percentiles - modified_cre))
                modified_cat_id = min_num_idx + modified_cre_val_idx
                sclaed_modified_cre = percentile_steps[modified_cre_val_idx]
                # Replace timeline & catalog ids
                modified_tl[:, -1, 5] = sclaed_modified_cre
                modified_ci[-1] = modified_cat_id

                # File name
                product_file_path = os.path.join(
                    adm_dir,
                    f"sim_cre_added_{cre_delta}_cre{modified_cre}_sex{sex}_{N_ITER}iter_horizon{HORIZON}.pkl",
                )

                if not os.path.exists(product_file_path):
                    yield modified_tl, modified_ci, N_ITER, horizon_start, timedelta(
                        days=HORIZON
                    ), product_file_path


def monte_carlo_to_all_covid(
    dataset_dir: str,
    output_dir: str,
    blueprint: str,
    max_gpus: int = None,
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
                blueprint=blueprint, output_dir=output_dir
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
