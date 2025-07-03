import os
import pickle
import traceback
import glob
from typing import Callable
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from uuid import uuid4
from concurrent.futures import as_completed, ProcessPoolExecutor
import psutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix  # Sparse matrix
from .general_ivds import GENERAL_IVDS
from ....general_params import watcher_config as config
from ....general_params.watcher_settings import BaseSettingsManager
from ....general_params.watcher_settings import get_settings
from ....utils import extract_numeric_from_text, load_numeric_stats

# Configs for evaluation sets
DSC = "discharge"
DEATH = "death"
CRE_SURGE = "serum_creatinine_surge"
NEUTROPENIA = "neutropenia"
AKI = "acute_kidney_injury"

TARGET_ATC_CODES = {
    # Broad-spectrum antibiotics
    "broad_spectrum_antibiotic_use": [
        [
            "J01CA12",  # piperacillin
            "J01CR05",  # piperacillin and beta-lactamase inhibitor
            "J01DE01",  # cefepime
            "J01DH",  # Carbapenems
            "J01GB12",  # arbekacin
            "J01XA",  # Glycopeptide antibiotics (including vancomycin and teicoplanin)
            "J01XB01",  # colistin
            "J01XX08",  # linezolid
            "J01XX09",  # daptomycin
        ],
        1,
    ],
    "broad_spectrum_antibiotic_use_gram_negative": [
        [
            "J01CA12",  # piperacillin
            "J01CR05",  # piperacillin and beta-lactamase inhibitor
            "J01DE01",  # cefepime
            "J01DH",  # Carbapenems
            "J01GB12",  # arbekacin
            "J01XB01",  # colistin
        ],
        1,
    ],
    "broad_spectrum_antibiotic_use_gram_positive": [
        [
            "J01GB12",  # arbekacin
            "J01XA",  # Glycopeptide antibiotics (including vancomycin and teicoplanin)
            "J01XX08",  # linezolid
            "J01XX09",  # daptomycin
        ],
        1,
    ],
    "anti-MRSA": [
        [
            "J01XA",  # Glycopeptide antibiotics (including vancomycin and teicoplanin)
            "J01XX08",  # linezolid
            "J01XX09",  # daptomycin
        ],
        1,
    ],
    "glycopeptids": [
        [
            "J01XA",  # Glycopeptide antibiotics (including vancomycin and teicoplanin)
        ],
        1,
    ],
    "carbapenems": [
        [
            "J01DH",  # Carbapenems
        ],
        1,
    ],
    # RBC transfusion
    "RBC_transfusion": [["B05AX01"], 1],
    # PLT transfusion
    "PLT_transfusion": [["B05AX02"], 1],
    # FFP transfusion
    "FFP_transfusion": [["B05AA02"], 1],
    # Catecolamine use (any)
    "catecholamine_use": [["C01CA"], 1],
    # Catecolamine use (>=2 days)
    "catecholamine_use_gte_2": [["C01CA"], 2],
}

TARGET_LAB_CODES = {
    # Sodium
    "serum_sodium": {
        "code": "3H0100000023---01",
        "unit": "mmol/L",
        "lower_thresholds": [120, 125, 130, 135],
        "upper_thresholds": [145, 150, 155, 160],
    },
    # Potassium
    "serum_potassium": {
        "code": "3H0150000023---01",
        "unit": "mmol/L",
        "lower_thresholds": [2.0, 2.5, 3.0, 3.5],
        "upper_thresholds": [5.0, 5.5, 6.0, 6.5, 7.0],
    },
    # Chloride
    "serum_chloride": {
        "code": "3H0200000023---01",
        "unit": "mmol/L",
        "lower_thresholds": [85, 90, 96],
        "upper_thresholds": [106, 110, 115, 120],
    },
}

# TODO: Implement AKI eval
# KDIGO criteria (summary):Increase in ≥0.3 mg/dL within 48 hours or ≥50% within 7 days
AKI_SETS = {
    CRE_SURGE: {
        "code": "3C0150000023---01",
        "unit": "mg/dL",
        "absolute_thresholds": [0.3],  # Within 48 hours
        "relative_thresholds": [1.5],  # Within 7 days
    },
}


# Neutropenia
# ANC = WBC * neutrophil fraction
NEUTROPENIA_SETS = {
    "wbc_code": "2A9900000019---52",  # CBC (complete blood count), whole blood(with additive), white blood cell count
    "neutrophil_code": [
        "2A1600000019---51",  # whole blood result, unit = %
        "2A1600000034---51",  # blood smear result, unit = %. This one is much less frequent.
    ],
    "wbc_unit": "×10^3/μL",
    "wbc_const": 1000,
    "thresholds": [100, 500],
}

ATC_RECORD_TYPES = [config.RECORD_TYPE_NUMBERS[t] for t in [config.PSC_O, config.INJ_O]]


def _select_codes(df: pd.DataFrame, percent: float = 90) -> list[str]:
    """Select top N codes based on percent appearing in the dataset."""
    df = df.sort_values("count", ascending=False)
    threshold = df["count"].sum() * percent / 100
    df["cumsum"] = df["count"].cumsum()
    filtered_labs = df[df["cumsum"] <= threshold]
    selected_codes = filtered_labs[config.COL_ITEM_CODE].unique().tolist()
    return selected_codes


def _count_codes(
    df: pd.DataFrame,
    target_dx_codes: list,
    target_med_codes: list,
    target_lab_codes: list,
) -> dict:
    """Tally medical codes into per-patient total and per-code count matrices.

    Returns a dict with keys 'diagnosis', 'drug', 'lab'. Each value is a dict with:
    - 'total': 1D array of total codes per patient (length = n_patients)
    - 'counts': 2D array of code counts per patient (shape = [n_patients, n_codes])
    """

    counts = {}
    type_map = {"diagnosis": [3], "drug": [4, 5], "lab": [6]}

    # Get all unique patient IDs
    all_patient_ids = df["patient_id"].unique()

    for category, type_values in type_map.items():
        # Count all codes
        type_mask = df["type"].isin(type_values)
        target_type_df = df.loc[df["type"].isin(type_values)]
        all_code_counts = target_type_df.groupby("patient_id").size()
        all_code_counts = all_code_counts.reindex(index=all_patient_ids, fill_value=0)
        all_code_counts_np = all_code_counts.values

        # Slice
        if category == "diagnosis":
            expected_codes = target_dx_codes
        elif category == "drug":
            expected_codes = target_med_codes
        else:
            expected_codes = target_lab_codes
        type_mask = df["type"].isin(type_values)
        subset_mask = df["code"].isin(expected_codes)
        subset = df[type_mask & subset_mask]

        # Count
        if subset.empty:
            # All-zero array
            counts_np = np.zeros(
                shape=(len(all_patient_ids), len(expected_codes))
            ).astype(np.uint8)
        else:
            # Pivot table (groupby patient_id and code counts)
            pivot_table = (
                subset.groupby(["patient_id", "code"]).size().unstack(fill_value=0)
            )
            # Ensure all expected codes are columns, in order
            pivot_table = pivot_table.reindex(columns=expected_codes, fill_value=0)

            # Ensure all patients are included
            pivot_table = pivot_table.reindex(index=all_patient_ids, fill_value=0)
            pivot_table = pivot_table.reset_index()

            # Convert data to matrix
            counts_np = pivot_table.drop("patient_id", axis=1).values
        counts_sparse = csr_matrix(counts_np)  # Sparse matrix for memory efficiency.

        counts[category] = {"total": all_code_counts_np, "counts": counts_sparse}

    return counts


def _cooccurrence_lab_codes(
    df: pd.DataFrame,
    target_lab_codes: list,
) -> dict:
    """
    Create binary co-occurrence matrices for lab codes.

    - 'cooccurrence': 2D binary matrix [n_patients, n_codes], 1 if the code was present at least once.
    """

    all_patient_ids = df["patient_id"].unique()

    # Select target code list
    expected_codes = target_lab_codes

    # Filter records by type and code
    type_mask = df["type"].isin([6])
    code_mask = df["code"].isin(expected_codes)
    subset = df[type_mask & code_mask]

    if subset.empty:
        cooc_matrix = np.zeros(
            (len(all_patient_ids), len(expected_codes)), dtype=np.uint8
        )
    else:
        # Binary matrix: 1 if code is present at least once per patient
        pivot_table = (
            subset.groupby(["patient_id", "code"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
        )

        # Ensure all codes are columns (in expected order)
        pivot_table = pivot_table.reindex(columns=expected_codes, fill_value=0)

        # Ensure all patients are included (fill missing with 0s)
        pivot_table = pivot_table.reindex(index=all_patient_ids, fill_value=0)

        cooc_matrix = pivot_table.values.astype(bool).astype(np.uint8)

    return cooc_matrix


# LEGACY: The correlation metrics can be computed using data obtainded from _collect_pivot_tables_dist
def _collect_pivot_tables_corr(
    df: pd.DataFrame, selected_codes: list[str]
) -> np.ndarray:
    """
    Extract a patient × lab matrix (values only) for selected lab codes.

    Parameters:
        df (pd.DataFrame): Lab records with columns 'patient_id', 'age', 'code', 'result'.
        selected_codes (list[str]): List of lab codes to include.

    Returns:
        np.ndarray: 2D array of shape (n_samples, len(selected_codes)), filled with lab values or NaN.
    """
    # Filter and extract numeric values
    df = df[df["code"].isin(selected_codes)].copy()
    df["numeric"] = extract_numeric_from_text(df["result"])
    df = df.dropna(subset=["numeric"])
    # Remove duplicates per (patient_id, age, code)
    df = df.drop_duplicates(subset=["patient_id", "age", "code"])
    # Pivot to wide format
    pv = df.pivot(index=["patient_id", "age"], columns="code", values="numeric")
    # Ensure all selected codes are present
    pv = pv.reindex(columns=selected_codes)
    pv = pv.values.astype(np.float32)

    return pv


def _collect_pivot_tables_dist(
    df: pd.DataFrame, selected_codes: list[str]
) -> np.ndarray:
    """
    Extract a patient × lab matrix (values only) for selected lab codes.
    """
    # NOTE: This function can cover _collect_pivot_tables_corr.
    # Get all simulation no
    all_sim_no = df["patient_id"].unique()
    # Filter and extract numeric values
    df = df[df["code"].isin(selected_codes)].copy()
    df["numeric"] = extract_numeric_from_text(df["result"])
    df = df.dropna(subset=["numeric"])
    # Remove duplicates per (patient_id, age, code)
    df = df.drop_duplicates(subset=["patient_id", "age", "code"])
    # Pivot to wide format
    pv = df.pivot(index=["patient_id", "age"], columns="code", values="numeric")
    # Ensure all selected codes are present
    pv = pv.reindex(columns=selected_codes)
    # Reset index and sort columns
    pv = pv.reset_index()
    pv = pv[["patient_id"] + selected_codes]
    # Ensure all patients (simulations) are included
    missing_ids = set(all_sim_no) - set(pv["patient_id"].unique())
    if missing_ids:
        filler = pd.DataFrame(
            {
                "patient_id": list(missing_ids),
                **{code: np.nan for code in selected_codes},
            }
        )
        pv = pd.concat([pv, filler], ignore_index=True)
    # Finalize
    pv = pv.sort_values(by="patient_id")
    pv[selected_codes] = pv[selected_codes].astype(
        np.float32
    )  # Convert to float32 for memory efficiency
    # This returned pv has the columns of 'patient_id' + selected_codes
    return pv


# Count length
def _count_tokens(df):
    """Count timeline length."""
    # Number of tokens
    non_demog = df.loc[df["type"] != 0]  # Exclude demographic data
    n_temp = int(non_demog["age"].nunique())
    n_code_or_token = int(non_demog["code"].replace("", pd.NA).notna().sum())
    n_results = int(non_demog["result"].replace("", pd.NA).notna().sum())
    n_tokens = n_temp + n_code_or_token + n_results

    return n_tokens


def _compute_length_calibration(
    df: pd.DataFrame, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        df (pd.DataFrame): DataFrame with columns 'patient_id', 'age', 'code', 'result'.
            Must contain patient_id and age columns.
        n_bins (int): Number of bins to create for length calibration.
            NOTE: If you want 20 spaces (0, 5, 10, ..., 100), then you need to set n_bins=21, not 20!.

    Returns:
        tuple: Lower and upper bounds of the bins for length calibration.
            Each is a 1D numpy array of shape (n_bins,).
            The lower bounds are percentiles from 0 to 50, and upper bounds from 100 to 50.
            The length of each array is n_bins.
    """
    # Set percentiles.
    lower_percentiles = np.linspace(50, 0, n_bins)
    upper_percentiles = np.linspace(50, 100, n_bins)
    # Get all simulated lengths
    all_lengths = df.groupby("patient_id").apply(_count_tokens).values
    # Create bins
    lower_bounds = np.percentile(all_lengths, lower_percentiles)
    upper_bounds = np.percentile(all_lengths, upper_percentiles)
    upper_bounds[
        0
    ] -= 0.01  # By subtracting this noise, it ensures the first bin (0% CI) does not include anything.

    return lower_bounds, upper_bounds


def _extract_lab_values(df: pd.DataFrame, target_code: str, unit: str) -> pd.DataFrame:
    """
    Extracts numeric values of a specific laboratory test from a DataFrame,
    filtering by target lab code and expected unit suffix.

    Args:
        df (pd.DataFrame): Input lab result table.
            Must contain columns: config.COL_CODE, config.COL_RESULT,
            config.COL_PID, config.COL_AGE.
        target_code (str): Lab test code to extract (e.g., 'Na', 'CRP').
        unit (str): Expected string suffix in the result (e.g., 'mg/dL').

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns ['patient_id', 'age', 'numeric'],
        where 'numeric' contains parsed float values.
    """
    # Filter by lab code and expected unit
    lab_mask = (df[config.COL_CODE] == target_code) & (
        df[config.COL_RESULT].fillna("").str.endswith(unit)
    )
    df_labs = df.loc[
        lab_mask, [config.COL_PID, config.COL_AGE, config.COL_RESULT]
    ].copy()

    # Extract numeric values
    df_labs["numeric"] = extract_numeric_from_text(df_labs[config.COL_RESULT])

    # Drop invalid rows
    df_labs = df_labs.dropna(subset=["numeric"])

    # Return clean table
    return df_labs[[config.COL_PID, config.COL_AGE, "numeric"]]


def _compute_anc(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate absolute nutrophil counts using wbc and neutrophil fractions.
    If duplicated counts are found, lower ones are selected.
    Args:
        df (pd.DataFrame): Input table.
    Returns:
        ancs (pd.DataFrame): Output table with absolute neutrophil counts.
            This table only contains three columns: patient ID, age and anc.
            (* anc for absolute neutrophil counts.)
    """
    wbc_code = NEUTROPENIA_SETS["wbc_code"]
    wbc_unit = NEUTROPENIA_SETS["wbc_unit"]
    wbc_const = NEUTROPENIA_SETS["wbc_const"]
    neut_codes = NEUTROPENIA_SETS["neutrophil_code"]
    # Select WBCs
    wbcs = df.loc[
        (df[config.COL_CODE] == wbc_code)
        & (df[config.COL_RESULT].str.endswith(wbc_unit))
    ].copy()
    wbcs["numeric"] = wbcs[config.COL_RESULT].str.extract(r"(\d+\.?\d*)", expand=True)
    wbcs["numeric"] = pd.to_numeric(wbcs["numeric"], errors="coerce")
    wbcs = wbcs.loc[~wbcs["numeric"].isna()]
    wbcs["numeric"] = wbcs["numeric"] * wbc_const
    wbcs = wbcs.rename(columns={"numeric": "wbc"})
    wbcs = wbcs[[config.COL_PID, config.COL_AGE, "wbc"]]
    # Select neutrophil fractions
    neuts = df.loc[
        (df[config.COL_CODE].isin(neut_codes))
        & (df[config.COL_RESULT].str.endswith("%"))
    ].copy()
    neuts["numeric"] = neuts[config.COL_RESULT].str.extract(r"(\d+\.?\d*)", expand=True)
    neuts["numeric"] = pd.to_numeric(neuts["numeric"], errors="coerce")
    neuts = neuts.loc[~neuts["numeric"].isna()]
    neuts["numeric"] = neuts["numeric"] * 0.01  # % -> fraction
    neuts = neuts.rename(columns={"numeric": "neut"})
    neuts = neuts[[config.COL_PID, config.COL_AGE, "neut"]]
    # Compute ANC
    ancs = pd.merge(wbcs, neuts, how="left", on=[config.COL_PID, config.COL_AGE])
    ancs["anc"] = ancs["wbc"] * ancs["neut"]
    ancs = ancs[[config.COL_PID, config.COL_AGE, "anc"]]
    # Sort and drop duplicates (* Latest value comes first)
    ancs = ancs.sort_values(
        [config.COL_PID, config.COL_AGE, "anc"], ascending=[True, False, True]
    )
    ancs = ancs.drop_duplicates([config.COL_PID, config.COL_AGE], keep="first")
    # Drop missing ANC (either wbc or neutrofil fraction missing)
    ancs = ancs.dropna(subset=["anc"])

    return ancs


def _eval_mc_adm_results(
    patient_dir: str,
    time_of_eval_td: timedelta,
    max_horizon_td: timedelta,
    max_days: int,
    fn_future: Callable,
    fn_past: Callable = None,
    kwargs_future: dict = None,
    kwargs_past: dict = None,
):
    # Load the full timelineand apply a custom function if indicated
    patient_no = patient_dir.split("/")[-1]
    total_stats = {}
    full_timeline = pd.read_pickle(os.path.join(patient_dir, "full_timeline.pkl"))
    admissions = [d for d in os.listdir(patient_dir) if d.startswith("admission")]
    for adm in admissions:
        # Load the time of admission and discharge
        with open(os.path.join(patient_dir, adm, "admission_info.pkl"), "rb") as f:
            info_data = pickle.load(f)
        time_adm, time_dsc, timestamp_adm, dept_adm = info_data
        date_adm = timedelta(days=time_adm.days)
        date_dsc = timedelta(days=time_dsc.days)
        n_days = (date_dsc - date_adm).days
        n_days = min(max_days, n_days)
        # Read through the simulation results per day
        total_stats[adm] = {}
        for day in range(0, n_days + 1):
            eval_time = date_adm + timedelta(days=day) + time_of_eval_td
            eval_end_time = eval_time + max_horizon_td
            eval_actual_timestamp = (
                datetime(
                    year=timestamp_adm.year,
                    month=timestamp_adm.month,
                    day=timestamp_adm.day,
                )
                + timedelta(days=day)
                + time_of_eval_td
            )
            if (time_adm <= eval_time) & (eval_time <= time_dsc):
                try:
                    # Open a simulation result
                    sim_result_path = os.path.join(patient_dir, adm, f"day{day}.pkl")
                    sim_result = pd.read_pickle(sim_result_path)
                    # Get the patient timeline before the time of evaluation
                    past_data = full_timeline.loc[
                        full_timeline[config.COL_AGE] <= eval_time
                    ]
                    # Slice out the actual patient timeline during the evaluation period
                    # NOTE: Preserving first demographic rows
                    dmg_mask_ft = full_timeline[config.COL_TYPE] == 0
                    max_horizon_mask_ft = (
                        full_timeline[config.COL_AGE] > eval_time
                    ) & (full_timeline[config.COL_AGE] <= eval_end_time)
                    future_data = full_timeline.loc[
                        max_horizon_mask_ft | dmg_mask_ft
                    ].copy()
                    # Slice out the simulation results
                    dmg_mask_sim = sim_result[config.COL_TYPE] == 0
                    max_horizon_mask_sim = (sim_result[config.COL_AGE] > eval_time) & (
                        sim_result[config.COL_AGE] <= eval_end_time
                    )
                    sim_result = sim_result.loc[max_horizon_mask_sim | dmg_mask_sim]

                    # Collect stats
                    n_sim = sim_result[config.COL_PID].nunique()
                    # *** DEBUG ***
                    if n_sim != 256:
                        print("Irregular n-sim encountered")
                        print("n-sim", n_sim)
                        print(sim_result_path)
                    # *************
                    if n_sim > 0:
                        if fn_past is not None:
                            if kwargs_past is not None:
                                past_stats = fn_past(
                                    past_data, eval_time, **kwargs_past
                                )
                            else:
                                past_stats = fn_past(past_data, eval_time)
                            if kwargs_future is not None:
                                future_stats = fn_future(
                                    future_data,
                                    sim_result,
                                    eval_time,
                                    max_horizon_td,
                                    eval_actual_timestamp,
                                    dept_adm,
                                    patient_no,
                                    past_stats,
                                    **kwargs_future,
                                )
                            else:
                                future_stats = fn_future(
                                    future_data,
                                    sim_result,
                                    eval_time,
                                    max_horizon_td,
                                    eval_actual_timestamp,
                                    dept_adm,
                                    patient_no,
                                    past_stats,
                                )
                        else:
                            if kwargs_future is not None:
                                future_stats = fn_future(
                                    future_data,
                                    sim_result,
                                    eval_time,
                                    max_horizon_td,
                                    **kwargs_future,
                                )
                            else:
                                future_stats = fn_future(
                                    future_data,
                                    sim_result,
                                    eval_time,
                                    max_horizon_td,
                                )
                    else:
                        future_stats = None
                    if future_stats is None:
                        print("Simulation stats were not obtained from", patient_dir)
                    else:
                        total_stats[adm][f"day{day}"] = future_stats

                except (FileNotFoundError, EOFError) as e:
                    print(f"Failed to open {sim_result_path}")
                    print("ERROR:", e)
                    print("*****TRACEBACK*******")
                    traceback.print_exc()
                    print("*********************")
                    continue

    return patient_no, total_stats


def eval_mc_adm_results(
    result_dir: str,
    fn_future: Callable,
    fn_past: Callable = None,
    max_workers: int = None,
    kwargs_future: dict = None,
    kwargs_past: dict = None,
):
    if max_workers is None:
        max_workers = psutil.cpu_count(logical=False) - 1
    patient_list = pd.read_pickle(os.path.join(result_dir, "patient_list.pkl"))
    if not result_dir.endswith("/"):
        result_dir = result_dir + "/"
    patient_list["path"] = result_dir + patient_list["path"]
    with open(os.path.join(result_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    time_of_eval = metadata["time_of_eval"]
    max_horizon = metadata["time_horizon"]
    max_days = metadata["max_days"]
    time_of_eval_td = timedelta(hours=time_of_eval)
    max_horizon_td = timedelta(days=max_horizon)

    # Debug handling
    if os.environ.get("DEBUG_MODE") == "1":
        debug_chunks = os.environ.get("DEBUG_CHUNKS", str(max_workers * 10))
        debug_chunks = int(debug_chunks)
        patient_list = patient_list.iloc[:debug_chunks]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = {}
        futures = [
            executor.submit(
                _eval_mc_adm_results,
                patient_dir=patient_dir,
                time_of_eval_td=time_of_eval_td,
                max_horizon_td=max_horizon_td,
                max_days=max_days,
                fn_future=fn_future,
                fn_past=fn_past,
                kwargs_future=kwargs_future,
                kwargs_past=kwargs_past,
            )
            for patient_dir in patient_list["path"]
        ]
        for future in tqdm(as_completed(futures), total=len(patient_list["path"])):
            patient_no, stats = future.result()
            if stats is not None:
                results[patient_no] = stats

    return results


# ********
# * Sets *
# ********
def _eval_mc_adm_set_fn_future(
    future_data: pd.DataFrame,
    sim_result: pd.DataFrame,
    eval_time: timedelta,
    max_horizon_td: timedelta,
    eval_actual_timestamp: datetime,
    dept_adm: str,
    patient_no: str,
    past_stats: tuple[dict],
):

    stats = {}
    latest_obs_dict, latest_vals_dict = past_stats
    n_sim = sim_result[config.COL_PID].nunique()
    if not dept_adm:
        dept_adm = ""

    if n_sim > 0:
        for horizon in range(1, max_horizon_td.days + 1):
            # Slice out the table for the desired period
            # NOTE: the dataframes contain demgraphic rows
            eval_end = eval_time + timedelta(days=horizon)
            ac_dmg_mask = future_data[config.COL_TYPE] == 0
            sim_dmg_mask = sim_result[config.COL_TYPE] == 0
            actual = future_data.loc[
                ac_dmg_mask | (future_data[config.COL_AGE] <= eval_end)
            ].copy()
            sim = sim_result.loc[
                sim_dmg_mask | (sim_result[config.COL_AGE] <= eval_end)
            ].copy()

            # ******************
            # * Clinical tasks *
            # ******************
            # Initialize
            patient_numbers = {}
            actual_stats = {}
            sim_stats = {}
            eval_timestamps = {}
            adm_depts = {}
            last_obs = {}
            last_vals = {}
            # 1. Discharge
            task_name = DSC
            actual_discharged = (actual[config.COL_CODE] == "[DSC]").any()
            n_discharges_in_sim = sim.loc[
                sim[config.COL_CODE] == "[DSC]", config.COL_PID
            ].nunique()
            sim_discharge_proba = n_discharges_in_sim / n_sim
            actual_stats[task_name] = int(actual_discharged)
            sim_stats[task_name] = sim_discharge_proba
            eval_timestamps[task_name] = eval_actual_timestamp
            adm_depts[task_name] = dept_adm
            patient_numbers[task_name] = patient_no
            last_obs[task_name] = None
            last_vals[task_name] = None

            # 1. In-hospital death
            task_name = DEATH
            actual_died = (actual[config.COL_RESULT] == "[DSC_EXP]").any()
            n_deaths_in_sim = sim.loc[
                sim[config.COL_RESULT] == "[DSC_EXP]", config.COL_PID
            ].nunique()
            sim_death_proba = n_deaths_in_sim / n_sim
            actual_stats[task_name] = int(actual_died)
            sim_stats[task_name] = sim_death_proba
            eval_timestamps[task_name] = eval_actual_timestamp
            adm_depts[task_name] = dept_adm
            patient_numbers[task_name] = patient_no
            last_obs[task_name] = None
            last_vals[task_name] = None

            # 3. Medication orders
            for task_name, code_info in TARGET_ATC_CODES.items():
                target_med_code, n_order_days = code_info
                actual_ordered = False
                actual_med_code_type_mask = actual[config.COL_TYPE].isin(
                    ATC_RECORD_TYPES
                )
                sim_med_code_type_mask = sim[config.COL_TYPE].isin(ATC_RECORD_TYPES)
                sim_ordered_mask = pd.Series(np.full(len(sim), False), index=sim.index)
                for c in target_med_code:
                    sim_ordered_mask = sim_ordered_mask | sim[
                        config.COL_CODE
                    ].str.startswith(c)
                sim_ordered_mask = sim_ordered_mask & sim_med_code_type_mask
                # Orders at least once
                if n_order_days == 1:
                    for c in target_med_code:
                        actual_ordered = (
                            actual[config.COL_CODE].str.startswith(c)
                            & actual_med_code_type_mask
                        ).any()
                        if actual_ordered:
                            break
                    n_ordered_in_sim = sim.loc[
                        sim_ordered_mask, config.COL_PID
                    ].nunique()
                    sim_ordered_proba = n_ordered_in_sim / n_sim
                # Orders >= 2 days
                else:
                    # Actual
                    actual_ordered_mask = pd.Series(
                        np.full(len(actual), False), index=actual.index
                    )
                    for c in target_med_code:
                        actual_ordered_mask = actual_ordered_mask | actual[
                            config.COL_CODE
                        ].str.startswith(c)
                    actual_ordered_mask = (
                        actual_ordered_mask & actual_med_code_type_mask
                    )
                    if actual_ordered_mask.any():
                        actual_ordered_days = actual.loc[
                            actual_ordered_mask, config.COL_AGE
                        ].dt.days.nunique()
                        actual_ordered = actual_ordered_days >= n_order_days
                    else:
                        actual_ordered = False
                    # Sim
                    sim[config.COL_DAYS] = sim[config.COL_AGE].dt.days
                    ordered_dates_in_sim = (
                        sim.loc[sim_ordered_mask, :]
                        .groupby(config.COL_PID)[config.COL_DAYS]
                        .nunique()
                    )
                    n_ordered_in_sim = (ordered_dates_in_sim >= n_order_days).sum()
                    sim = sim.drop(config.COL_DAYS, axis=1)

                sim_ordered_proba = n_ordered_in_sim / n_sim
                actual_stats[task_name] = int(actual_ordered)
                sim_stats[task_name] = sim_ordered_proba
                eval_timestamps[task_name] = eval_actual_timestamp
                adm_depts[task_name] = dept_adm
                patient_numbers[task_name] = patient_no
                last_obs[task_name] = latest_obs_dict[task_name]
                last_vals[task_name] = None
            # 4. Laboratory test results
            for task_name, lab_dict in TARGET_LAB_CODES.items():
                target_code = lab_dict[config.COL_CODE]
                unit = lab_dict["unit"]
                low_thresholds = lab_dict["lower_thresholds"]
                up_thresholds = lab_dict["upper_thresholds"]
                last_lab_obs = latest_obs_dict[task_name]
                last_hypo_obs = last_lab_obs[0]
                last_hyper_obs = last_lab_obs[1]
                last_lab_val = latest_vals_dict[task_name]
                actual_labs = _extract_lab_values(actual, target_code, unit)
                sim_labs = _extract_lab_values(sim, target_code, unit)
                # Hypo
                for j, t in enumerate(low_thresholds):
                    if actual_labs.size:
                        actual_hypo_seen = (actual_labs["numeric"] < t).any()
                    else:
                        actual_hypo_seen = False
                    if sim_labs.size:
                        n_hypo_in_sim = sim_labs.loc[
                            sim_labs["numeric"] < t, config.COL_PID
                        ].nunique()
                    else:
                        n_hypo_in_sim = 0
                    sim_hypo_proba = n_hypo_in_sim / n_sim
                    key = f"{task_name}_<_{t}_{unit}"
                    actual_stats[key] = int(actual_hypo_seen)
                    sim_stats[key] = sim_hypo_proba
                    eval_timestamps[key] = eval_actual_timestamp
                    adm_depts[key] = dept_adm
                    patient_numbers[key] = patient_no
                    last_obs[key] = last_hypo_obs[j]
                    last_vals[key] = last_lab_val
                # Hyper
                for j, t in enumerate(up_thresholds):
                    if actual_labs.size:
                        actual_hyper_seen = (actual_labs["numeric"] > t).any()
                    else:
                        actual_hyper_seen = False
                    if sim_labs.size:
                        n_hyper_in_sim = sim_labs.loc[
                            sim_labs["numeric"] > t, config.COL_PID
                        ].nunique()
                    else:
                        n_hyper_in_sim = 0
                    sim_hyper_proba = n_hyper_in_sim / n_sim
                    key = f"{task_name}_>_{t}_{unit}"
                    actual_stats[key] = int(actual_hyper_seen)
                    sim_stats[key] = sim_hyper_proba
                    eval_timestamps[key] = eval_actual_timestamp
                    adm_depts[key] = dept_adm
                    patient_numbers[key] = patient_no
                    last_obs[key] = last_hyper_obs[j]
                    last_vals[key] = last_lab_val
            # 5. AKI
            task_name = CRE_SURGE
            base_cre = latest_vals_dict[task_name]
            aki_dict = AKI_SETS[task_name]
            target_code = aki_dict[config.COL_CODE]
            unit = aki_dict["unit"]
            abs_thresholds = aki_dict["absolute_thresholds"]
            rel_thresholds = aki_dict["relative_thresholds"]
            actual_cres = _extract_lab_values(actual, target_code, unit)
            sim_cres = _extract_lab_values(sim, target_code, unit)
            # Exclude cases without baseline creatinine
            if base_cre is not None:
                # Absolute creatinine increase
                for increment in abs_thresholds:
                    threshold = base_cre + increment
                    if actual_cres.size:
                        actual_cre_seen = (actual_cres["numeric"] >= threshold).any()
                    else:
                        actual_cre_seen = False
                    if sim_cres.size:
                        n_cre_in_sim = sim_cres.loc[
                            sim_cres["numeric"] >= threshold, config.COL_PID
                        ].nunique()
                    else:
                        n_cre_in_sim = 0
                    sim_cre_proba = n_cre_in_sim / n_sim
                    key = f"{task_name}_+_{increment}_{unit}"
                    actual_stats[key] = int(actual_cre_seen)
                    sim_stats[key] = sim_cre_proba
                    eval_timestamps[key] = eval_actual_timestamp
                    adm_depts[key] = dept_adm
                    patient_numbers[key] = patient_no
                    last_obs[key] = None
                    last_vals[key] = base_cre
                # Relative creatinine increase
                for rate in rel_thresholds:
                    threshold = base_cre * rate
                    if actual_cres.size:
                        actual_cre_seen = (actual_cres["numeric"] >= threshold).any()
                    else:
                        actual_cre_seen = False
                    if sim_cres.size:
                        n_cre_in_sim = sim_cres.loc[
                            sim_cres["numeric"] >= threshold, config.COL_PID
                        ].nunique()
                    else:
                        n_cre_in_sim = 0
                    sim_cre_proba = n_cre_in_sim / n_sim
                    key = f"{task_name}_*_{rate}"
                    actual_stats[key] = int(actual_cre_seen)
                    sim_stats[key] = sim_cre_proba
                    eval_timestamps[key] = eval_actual_timestamp
                    adm_depts[key] = dept_adm
                    patient_numbers[key] = patient_no
                    last_obs[key] = None
                    last_vals[key] = base_cre

                    # Combined AKI criteria
                    if (horizon == 7) and (rate == 1.5):
                        # 48hr criteria
                        threshold_48hr = base_cre + 0.3
                        after_48hr = eval_time + timedelta(days=2)
                        actual_cres_48hr = actual_cres[
                            actual_cres[config.COL_AGE] <= after_48hr
                        ]
                        sim_cres_48hr = sim_cres[sim_cres[config.COL_AGE] <= after_48hr]
                        # Actual AKI
                        if actual_cres_48hr.size:
                            actual_cre_seen_48hr = (
                                actual_cres_48hr["numeric"] >= threshold_48hr
                            ).any()
                            actual_aki_seen = actual_cre_seen or actual_cre_seen_48hr
                        else:
                            actual_aki_seen = actual_cre_seen
                        # AKI in simulation
                        if sim_cres_48hr.size:
                            sim_numbers_cre_48hr = set(
                                sim_cres_48hr.loc[
                                    sim_cres_48hr["numeric"] >= threshold_48hr,
                                    config.COL_PID,
                                ].unique()
                            )
                        else:
                            sim_numbers_cre_48hr = set()
                        if sim_cres.size:
                            sim_numbers_cre_7d = set(
                                sim_cres.loc[
                                    sim_cres["numeric"] >= threshold, config.COL_PID
                                ].unique()
                            )
                        else:
                            sim_numbers_cre_7d = set()
                        sim_numbers_aki = sim_numbers_cre_48hr | sim_numbers_cre_7d
                        n_aki_in_sim = len(sim_numbers_aki)
                        sim_aki_proba = n_aki_in_sim / n_sim
                        actual_stats[AKI] = int(actual_aki_seen)
                        sim_stats[AKI] = sim_aki_proba
                        eval_timestamps[AKI] = eval_actual_timestamp
                        adm_depts[AKI] = dept_adm
                        patient_numbers[AKI] = patient_no
                        last_obs[AKI] = None
                        last_vals[AKI] = base_cre

            # 6. Neutropenia
            task_name = NEUTROPENIA
            last_np_obs = latest_obs_dict[task_name]
            last_anc_val = latest_vals_dict[task_name]
            actual_ancs = _compute_anc(actual)
            sim_ancs = _compute_anc(sim)
            for j, t in enumerate(NEUTROPENIA_SETS["thresholds"]):
                if actual_ancs.size:
                    actual_np_seen = (actual_ancs["anc"] < t).any()
                else:
                    actual_np_seen = False
                if sim_ancs.size:
                    n_np_in_sim = sim_ancs.loc[
                        sim_ancs["anc"] < t, config.COL_PID
                    ].nunique()
                else:
                    n_np_in_sim = 0
                sim_np_proba = n_np_in_sim / n_sim
                key = f"{task_name}_<_{t}_/μL"
                actual_stats[key] = int(actual_np_seen)
                sim_stats[key] = sim_np_proba
                eval_timestamps[key] = eval_actual_timestamp
                adm_depts[key] = dept_adm
                patient_numbers[key] = patient_no
                last_obs[key] = last_np_obs[j]
                last_vals[key] = last_anc_val

            # ********************************
            # * Add results to the main dict *
            # ********************************
            stats[f"{horizon}_days_prediction"] = {
                "actual_event": actual_stats,
                "simulation_event": sim_stats,
                "timestamp": eval_timestamps,
                "department": adm_depts,
                "patient_number": patient_numbers,
                "last_observation": last_obs,
                "last_value": last_vals,
            }

    # When n_sim ==0, avoid ZeroDivisionError
    else:
        print("WARNING!, zero simulation results encountered")
        stats = None

    return stats


def _eval_mc_adm_set_fn_past(
    past_data,
    eval_time,
) -> list:
    """Collects time of past events."""
    # Initialize lists
    latest_obs_dict = {}  # The latest time a target event is observed
    latest_vals_dict = {}  # The latest value of the target laboratory value
    # 1. Discharges
    latest_obs_dict[DSC] = None
    latest_vals_dict[DSC] = None
    # 2. In-hospital death
    latest_obs_dict[DEATH] = None
    latest_vals_dict[DEATH] = None
    # 3. Medication orders (Find recent observations, looking back 7 days.)
    for task_name, code_info in TARGET_ATC_CODES.items():
        codes, _ = code_info
        df = past_data.loc[past_data[config.COL_AGE] >= eval_time - timedelta(days=7)]
        past_order_mask = pd.Series(np.full(len(df), False), index=df.index)
        for c in codes:
            # NOTE: Currently, fn_past does not care about n_order_days in the past
            past_order_mask = past_order_mask | df[config.COL_CODE].str.startswith(c)
        if past_order_mask.any():
            past_orders = df.loc[past_order_mask]
            latest_time_observed = (eval_time - past_orders[config.COL_AGE]).min()
            latest_time_observed = latest_time_observed.to_pytimedelta()
        else:
            latest_time_observed = None
        latest_obs_dict[task_name] = latest_time_observed
        latest_vals_dict[task_name] = None
    # 4. Laboratory test results (Find recent observations, looking back 7 days.)
    # NOTE: Elements for latest observations appended here are tuples. The first element is for hypo and the next one is for hyper.
    for task_name, lab_dict in TARGET_LAB_CODES.items():
        code = lab_dict[config.COL_CODE]
        unit = lab_dict["unit"]
        low_thresholds = lab_dict["lower_thresholds"]
        up_thresholds = lab_dict["upper_thresholds"]
        df = past_data.loc[
            past_data[config.COL_AGE] >= eval_time - timedelta(days=7)
        ].copy()
        lab_df = _extract_lab_values(df, code, unit)
        lab_df = lab_df.sort_values(config.COL_AGE, ascending=False)
        # Latest value
        if lab_df.size:
            latest_value = lab_df.loc[lab_df.index[0], "numeric"]
        else:
            latest_value = None
        # Last time hypo was observed
        hypo_obs = []
        for t in low_thresholds:
            already_seen_mask = lab_df["numeric"] < t
            if already_seen_mask.any():
                past_vals = lab_df.loc[already_seen_mask]
                latest_time_observed = (eval_time - past_vals[config.COL_AGE]).min()
                latest_time_observed = latest_time_observed.to_pytimedelta()
            else:
                latest_time_observed = None
            hypo_obs.append(latest_time_observed)
        # Last time hyper was observed
        hyper_obs = []
        for t in up_thresholds:
            already_seen_mask = lab_df["numeric"] > t
            if already_seen_mask.any():
                past_vals = lab_df.loc[already_seen_mask]
                latest_time_observed = (eval_time - past_vals[config.COL_AGE]).min()
                latest_time_observed = latest_time_observed.to_pytimedelta()
            else:
                latest_time_observed = None
            hyper_obs.append(latest_time_observed)
        latest_obs_dict[task_name] = [hypo_obs, hyper_obs]
        latest_vals_dict[task_name] = latest_value
    # 5. AKI (find the latest creatinine value)
    task_name = CRE_SURGE
    aki_dict = AKI_SETS[CRE_SURGE]
    code = aki_dict[config.COL_CODE]
    unit = aki_dict["unit"]
    # Find creatine measurements within the day
    df = past_data.loc[
        past_data[config.COL_AGE] >= timedelta(days=eval_time.days)
    ].copy()
    cre_df = _extract_lab_values(df, code, unit)
    cre_df = cre_df.sort_values(config.COL_AGE, ascending=False)
    if cre_df.size:
        latest_value = cre_df.loc[cre_df.index[0], "numeric"]
    else:
        latest_value = None
    latest_obs_dict[task_name] = None
    latest_vals_dict[task_name] = latest_value
    # 6. Neutropenia
    task_name = NEUTROPENIA
    df = past_data.loc[
        past_data[config.COL_AGE] >= eval_time - timedelta(days=7)
    ].copy()
    ancs = _compute_anc(df)
    if ancs.size:
        ancs = ancs.sort_values(config.COL_AGE, ascending=False)
        latest_value = ancs.loc[ancs.index[0], "anc"]
    neutropenia_obs = []
    for t in NEUTROPENIA_SETS["thresholds"]:
        already_seen_mask = ancs["anc"] < t
        if already_seen_mask.any():
            past_vals = ancs.loc[already_seen_mask]
            latest_time_observed = (eval_time - past_vals[config.COL_AGE]).min()
            latest_time_observed = latest_time_observed.to_pytimedelta()
        else:
            latest_time_observed = None
        neutropenia_obs.append(latest_time_observed)
    latest_obs_dict[task_name] = neutropenia_obs
    latest_vals_dict[task_name] = latest_value

    # Finalize stats
    past_stats = (latest_obs_dict, latest_vals_dict)
    return past_stats


def eval_mc_adm_set(
    result_dir: str,
    max_workers: int = None,
):
    # Collect stats
    stats = eval_mc_adm_results(
        result_dir=result_dir,
        fn_future=_eval_mc_adm_set_fn_future,
        fn_past=_eval_mc_adm_set_fn_past,
        max_workers=max_workers,
    )

    # Tally stats
    print("Tally stats...")
    tallied_stats = {}
    for patient_no, admission_results in tqdm(stats.items()):
        for day_level_results in admission_results.values():
            for day, day_results in day_level_results.items():
                if day not in tallied_stats:
                    tallied_stats[day] = {}
                for horizon, hor_result in day_results.items():
                    # Tally clinical task performance
                    if horizon not in tallied_stats[day]:
                        tallied_stats[day][horizon] = {}
                    horizon_level_dict = tallied_stats[day][horizon]
                    for hor_k, hor_v in hor_result.items():
                        if hor_k not in horizon_level_dict:
                            horizon_level_dict[hor_k] = {}
                        for k, v in hor_v.items():
                            if k not in horizon_level_dict[hor_k]:
                                horizon_level_dict[hor_k][k] = []
                            horizon_level_dict[hor_k][k].append(v)

    return tallied_stats


# **********
# * Counts *
# **********
def _eval_mc_adm_count_fn_future(
    future_data: pd.DataFrame,
    sim_result: pd.DataFrame,
    eval_time: timedelta,
    max_horizon_td: timedelta,
    target_dx_codes: list,
    target_med_codes: list,
    target_lab_codes: list,
):

    stats = {}
    n_sim = sim_result[config.COL_PID].nunique()
    if n_sim > 0:
        for horizon in range(1, max_horizon_td.days + 1):
            # Slice out the table for the desired period
            # NOTE: the dataframes contain demgraphic rows
            eval_end = eval_time + timedelta(days=horizon)
            ac_dmg_mask = future_data[config.COL_TYPE] == 0
            sim_dmg_mask = sim_result[config.COL_TYPE] == 0
            actual = future_data.loc[
                ac_dmg_mask | (future_data[config.COL_AGE] <= eval_end)
            ].copy()
            sim = sim_result.loc[
                sim_dmg_mask | (sim_result[config.COL_AGE] <= eval_end)
            ].copy()

            # **********
            # * Counts *
            # **********
            codes_kwargs = {
                "target_dx_codes": target_dx_codes,
                "target_med_codes": target_med_codes,
                "target_lab_codes": target_lab_codes,
            }
            # Code counts
            actual_code_counts = _count_codes(actual, **codes_kwargs)
            sim_code_counts = _count_codes(sim, **codes_kwargs)

            # ********************************
            # * Add results to the main dict *
            # ********************************
            stats[f"{horizon}_days"] = {
                "actual_code_counts": actual_code_counts,
                "simulation_code_counts": sim_code_counts,
            }

    # When n_sim ==0, avoid ZeroDivisionError
    else:
        print("WARNING!, zero simulation results encountered")
        stats = None

    return stats


def eval_mc_adm_count(
    result_dir: str,
    dataset_dir: str,
    code_percentile: int,
    max_workers: int = None,
):
    # Select target codes
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        debug=os.environ.get("DEBUG_MODE") == "1",
        debug_chunks=int(os.environ.get("DEBUG_CHUNKS", "10")),
    )
    settings_manager.write()
    try:
        dtypes = {"original_value": str, "train_ID_and_train_period": int}
        dx_code_counts_df = pd.read_csv(
            get_settings("DX_CODE_COUNTS_PTH"),
            usecols=list(dtypes.keys()),
            dtype=dtypes,
        ).sort_values("train_ID_and_train_period", ascending=False)
        med_code_counts_df = pd.read_csv(
            get_settings("MED_CODE_COUNTS_PTH"),
            usecols=list(dtypes.keys()),
            dtype=dtypes,
        ).sort_values("train_ID_and_train_period", ascending=False)
        lab_code_counts_df = pd.read_csv(
            get_settings("LAB_CODE_COUNTS_PTH"),
            usecols=list(dtypes.keys()),
            dtype=dtypes,
        ).sort_values("train_ID_and_train_period", ascending=False)
        # Rename
        col_mapper = {
            "original_value": config.COL_ITEM_CODE,
            "train_ID_and_train_period": "count",
        }
        dx_code_counts_df.rename(columns=col_mapper, inplace=True)
        med_code_counts_df.rename(columns=col_mapper, inplace=True)
        lab_code_counts_df.rename(columns=col_mapper, inplace=True)
        # Select codes
        target_dx_codes = _select_codes(
            dx_code_counts_df[[config.COL_ITEM_CODE, "count"]], percent=code_percentile
        )
        target_med_codes = _select_codes(
            med_code_counts_df[[config.COL_ITEM_CODE, "count"]], percent=code_percentile
        )
        target_lab_codes = _select_codes(
            lab_code_counts_df[[config.COL_ITEM_CODE, "count"]], percent=code_percentile
        )
        del med_code_counts_df, lab_code_counts_df, dx_code_counts_df
        print("N dx codes;", len(target_dx_codes))
        print("N med codes;", len(target_med_codes))
        print("N lab codes;", len(target_lab_codes))

        # Collect stats
        codes_kwargs = {
            "target_dx_codes": target_dx_codes,
            "target_med_codes": target_med_codes,
            "target_lab_codes": target_lab_codes,
        }
        stats = eval_mc_adm_results(
            result_dir=result_dir,
            fn_future=_eval_mc_adm_count_fn_future,
            fn_past=None,
            max_workers=max_workers,
            kwargs_future=codes_kwargs,
        )

        # Tally stats
        print("Tally stats...")
        tallied_stats = {}
        for patient_no, admission_results in tqdm(stats.items()):
            for day_level_results in admission_results.values():
                for day, day_results in day_level_results.items():
                    if day not in tallied_stats:
                        tallied_stats[day] = {}
                    for horizon, hor_result in day_results.items():
                        if horizon not in tallied_stats[day]:
                            tallied_stats[day][horizon] = {
                                "actual_code_counts": {
                                    "diagnosis": [],
                                    "drug": [],
                                    "lab": [],
                                },
                                "simulation_code_counts": {
                                    "diagnosis": [],
                                    "drug": [],
                                    "lab": [],
                                },
                            }
                        # Code counts
                        for code_type in ["diagnosis", "drug", "lab"]:
                            # Actual counts (np.ndarray)
                            ac = hor_result["actual_code_counts"][code_type]
                            tallied_stats[day][horizon]["actual_code_counts"][
                                code_type
                            ].append(ac)
                            # Simulation counts (DataFrame)
                            sc = hor_result["simulation_code_counts"][code_type]
                            tallied_stats[day][horizon]["simulation_code_counts"][
                                code_type
                            ].append(sc)
        # Add codes
        tallied_stats["target_dx_codes"] = target_dx_codes
        tallied_stats["target_med_codes"] = target_med_codes
        tallied_stats["target_lab_codes"] = target_lab_codes

        return tallied_stats

    finally:
        settings_manager.delete()


# ***************
# * Coocurrence *
# ***************
def _eval_mc_adm_lab_cooc_fn_future(
    future_data: pd.DataFrame,
    sim_result: pd.DataFrame,
    eval_time: timedelta,
    max_horizon_td: timedelta,
    target_lab_codes: list,
):

    stats = {}
    n_sim = sim_result[config.COL_PID].nunique()
    if n_sim > 0:
        for horizon in range(1, max_horizon_td.days + 1):
            # Slice out the table for the desired period
            # NOTE: the dataframes contain demgraphic rows
            eval_end = eval_time + timedelta(days=horizon)
            ac_dmg_mask = future_data[config.COL_TYPE] == 0
            sim_dmg_mask = sim_result[config.COL_TYPE] == 0
            actual = future_data.loc[
                ac_dmg_mask | (future_data[config.COL_AGE] <= eval_end)
            ].copy()
            sim = sim_result.loc[
                sim_dmg_mask | (sim_result[config.COL_AGE] <= eval_end)
            ].copy()

            # ****************
            # * Coocurrences *
            # ****************
            # Code counts
            actual_cooc = _cooccurrence_lab_codes(
                actual, target_lab_codes=target_lab_codes
            )
            sim_cooc = _cooccurrence_lab_codes(sim, target_lab_codes=target_lab_codes)

            # ********************************
            # * Add results to the main dict *
            # ********************************
            stats[f"{horizon}_days"] = {
                "actual_cooc": actual_cooc,
                "simulation_cooc": sim_cooc,
            }

    # When n_sim ==0, avoid ZeroDivisionError
    else:
        print("WARNING!, zero simulation results encountered")
        stats = None

    return stats


def eval_mc_adm_lab_cooc(
    result_dir: str,
    dataset_dir: str,
    code_percentile: int = 90,
    max_workers: int = None,
):
    # Select target codes
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        debug=os.environ.get("DEBUG_MODE") == "1",
        debug_chunks=int(os.environ.get("DEBUG_CHUNKS", "10")),
    )
    settings_manager.write()
    try:
        dtypes = {"original_value": str, "train_ID_and_train_period": int}
        lab_code_counts_df = pd.read_csv(
            get_settings("LAB_CODE_COUNTS_PTH"),
            usecols=list(dtypes.keys()),
            dtype=dtypes,
        ).sort_values("train_ID_and_train_period", ascending=False)
        # Rename
        col_mapper = {
            "original_value": config.COL_ITEM_CODE,
            "train_ID_and_train_period": "count",
        }
        lab_code_counts_df.rename(columns=col_mapper, inplace=True)
        # Select codes
        target_lab_codes = _select_codes(
            lab_code_counts_df[[config.COL_ITEM_CODE, "count"]], percent=code_percentile
        )
        del lab_code_counts_df
        print("N lab codes;", len(target_lab_codes))

        # Collect stats
        stats = eval_mc_adm_results(
            result_dir=result_dir,
            fn_future=_eval_mc_adm_lab_cooc_fn_future,
            fn_past=None,
            max_workers=max_workers,
            kwargs_future=dict(
                target_lab_codes=target_lab_codes,
            ),
        )

        # Tally stats
        print("Tally stats...")
        tallied_stats = {}
        for patient_no, admission_results in tqdm(stats.items()):
            for day_level_results in admission_results.values():
                for day, day_results in day_level_results.items():
                    if day not in tallied_stats:
                        tallied_stats[day] = {}
                    for horizon, hor_result in day_results.items():
                        if horizon not in tallied_stats[day]:
                            tallied_stats[day][horizon] = {
                                "actual_cooc": [],
                                "simulation_cooc": [],
                            }
                        # Actual counts (np.ndarray)
                        ac = hor_result["actual_cooc"]
                        tallied_stats[day][horizon]["actual_cooc"].append(ac)
                        # Simulation counts (DataFrame)
                        sc = hor_result["simulation_cooc"]
                        tallied_stats[day][horizon]["simulation_cooc"].append(sc)

        # Add codes
        tallied_stats["codes"] = target_lab_codes

        return tallied_stats

    finally:
        settings_manager.delete()


# ********
# * Corr *
# ********
# NOTE: LEGACY: This paret is legacy. Correlations can be computed with the data obtained from the `eval_mc_adm_lab_dist' function.
#       Use that function instead of this.
def _eval_mc_adm_corr(
    patient_dir: str,
    selected_codes: list,
    time_of_eval_td: timedelta,
    max_horizon_td: timedelta,
    max_days: int,
):
    # Load the full timelineand apply a custom function if indicated
    patient_no = patient_dir.split("/")[-1]
    total_stats = {}
    full_timeline = pd.read_pickle(os.path.join(patient_dir, "full_timeline.pkl"))
    admissions = [d for d in os.listdir(patient_dir) if d.startswith("admission")]
    act_pivots = []
    sim_pivots = []
    for adm in admissions:
        # Load the time of admission and discharge
        with open(os.path.join(patient_dir, adm, "admission_info.pkl"), "rb") as f:
            info_data = pickle.load(f)
        time_adm, time_dsc, timestamp_adm, dept_adm = info_data
        date_adm = timedelta(days=time_adm.days)
        date_dsc = timedelta(days=time_dsc.days)
        n_days = (date_dsc - date_adm).days
        n_days = min(max_days, n_days)
        # Collect entire patient labs during the admission
        # NOTE: Collect all lab tests during an admission
        full_adm_mask = (full_timeline[config.COL_AGE] > time_adm) & (
            full_timeline[config.COL_AGE] <= time_dsc
        )
        full_adm = full_timeline.loc[full_adm_mask].copy()
        act_pv = _collect_pivot_tables_corr(full_adm, selected_codes=selected_codes)
        if act_pv.size > 0:
            act_pivots.append(act_pv)

        # Read through the simulation results per day
        total_stats[adm] = {}
        for day in range(0, n_days + 1):
            eval_time = date_adm + timedelta(days=day) + time_of_eval_td
            eval_end_time = eval_time + max_horizon_td
            if (time_adm <= eval_time) & (eval_time <= time_dsc):
                try:
                    # Open a simulation result
                    sim_result_path = os.path.join(patient_dir, adm, f"day{day}.pkl")
                    sim_result = pd.read_pickle(sim_result_path)
                    # Slice out the simulation results
                    dmg_mask_sim = sim_result[config.COL_TYPE] == 0
                    max_horizon_mask_sim = (sim_result[config.COL_AGE] > eval_time) & (
                        sim_result[config.COL_AGE] <= eval_end_time
                    )
                    sim_result = sim_result.loc[max_horizon_mask_sim | dmg_mask_sim]

                    # Collect stats
                    n_sim = sim_result[config.COL_PID].nunique()

                    # *** DEBUG ***
                    if n_sim != 256:
                        print("Irregular n-sim encountered")
                        print("n-sim", n_sim)
                        print(sim_result_path)
                    # *************
                    if n_sim > 0:
                        # NOTE: Collect all lab values in the entire simulation
                        sim_pv = _collect_pivot_tables_corr(
                            sim_result, selected_codes=selected_codes
                        )
                        if sim_pv.size:
                            sim_pivots.append(sim_pv)
                    else:
                        pass

                except (FileNotFoundError, EOFError) as e:
                    print(f"Failed to open {sim_result_path}")
                    print("ERROR:", e)
                    print("*****TRACEBACK*******")
                    traceback.print_exc()
                    print("*********************")
                    continue

    if act_pivots:
        act_pivots = np.concatenate(act_pivots, axis=0)
    else:
        act_pivots = np.array([])
    if sim_pivots:
        sim_pivots = np.concatenate(sim_pivots, axis=0)
    else:
        sim_pivots = np.array([])

    return patient_no, act_pivots, sim_pivots


def eval_mc_adm_corr(
    result_dir: str,
    dataset_dir: str,
    code_percentile: int = 80,
    max_workers: int | None = None,
):
    # Select target codes
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        debug=os.environ.get("DEBUG_MODE") == "1",
        debug_chunks=int(os.environ.get("DEBUG_CHUNKS", "10")),
    )
    settings_manager.write()
    try:
        # Select codes
        num_stats = load_numeric_stats(file_path=None, single_unit=True).sort_values(
            "count", ascending=False
        )
        selected_codes = _select_codes(
            df=num_stats[[config.COL_ITEM_CODE, "count"]], percent=code_percentile
        )
        # selected_codes = list(GENERAL_IVDS.values())
        if max_workers is None:
            max_workers = psutil.cpu_count(logical=False) - 1
        patient_list = pd.read_pickle(os.path.join(result_dir, "patient_list.pkl"))
        if not result_dir.endswith("/"):
            result_dir = result_dir + "/"
        patient_list["path"] = result_dir + patient_list["path"]
        with open(os.path.join(result_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        time_of_eval = metadata["time_of_eval"]
        max_horizon = metadata["time_horizon"]
        max_days = metadata["max_days"]
        time_of_eval_td = timedelta(hours=time_of_eval)
        max_horizon_td = timedelta(days=max_horizon)

        # Debug handling
        if os.environ.get("DEBUG_MODE") == "1":
            debug_chunks = os.environ.get("DEBUG_CHUNKS", str(max_workers * 10))
            debug_chunks = int(debug_chunks)
            patient_list = patient_list.iloc[:debug_chunks]

        all_act_pivots = []
        all_sim_pivots = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _eval_mc_adm_corr,
                    patient_dir=patient_dir,
                    selected_codes=selected_codes,
                    time_of_eval_td=time_of_eval_td,
                    max_horizon_td=max_horizon_td,
                    max_days=max_days,
                )
                for patient_dir in patient_list["path"]
            ]
            for future in tqdm(as_completed(futures), total=len(patient_list["path"])):
                patient_no, act_pivots, sim_pivots = future.result()
                if act_pivots.size > 0:
                    all_act_pivots.append(act_pivots)
                if sim_pivots.size > 0:
                    all_sim_pivots.append(sim_pivots)

        return all_act_pivots, all_sim_pivots, selected_codes

    finally:
        settings_manager.delete()


# ********
# * Dist *
# ********
def _eval_mc_adm_lab_dist(
    patient_dir: str,
    selected_codes: list,
    time_of_eval_td: timedelta,
    max_horizon: timedelta,
    max_days: int,
    temp_dir: str,
):
    # Load the full timelineand apply a custom function if indicated
    patient_no = patient_dir.split("/")[-1]
    total_stats = {}
    full_timeline = pd.read_pickle(os.path.join(patient_dir, "full_timeline.pkl"))
    admissions = [d for d in os.listdir(patient_dir) if d.startswith("admission")]
    act_pivots = {}
    sim_pivots = {}
    for adm in admissions:
        # Load the time of admission and discharge
        with open(os.path.join(patient_dir, adm, "admission_info.pkl"), "rb") as f:
            info_data = pickle.load(f)
        time_adm, time_dsc, timestamp_adm, dept_adm = info_data
        date_adm = timedelta(days=time_adm.days)
        date_dsc = timedelta(days=time_dsc.days)
        n_days = (date_dsc - date_adm).days
        n_days = min(max_days, n_days)
        # Slice full real timeline with decent pad
        # NOTE: Demographic rows are excluded here from the actual df.
        full_adm_mask = (full_timeline[config.COL_AGE] > time_adm) & (
            full_timeline[config.COL_AGE] <= (time_dsc + timedelta(days=max_horizon))
        )
        full_adm = full_timeline.loc[full_adm_mask].copy()

        # Read through the simulation results per day
        total_stats[adm] = {}
        for day in range(0, n_days + 1):
            eval_time = date_adm + timedelta(days=day) + time_of_eval_td
            if (time_adm <= eval_time) & (eval_time <= time_dsc):
                try:
                    # Open a simulation result
                    sim_result_path = os.path.join(patient_dir, adm, f"day{day}.pkl")
                    sim_result = pd.read_pickle(sim_result_path)
                    # Slice out the simulation results
                    dmg_mask_sim = sim_result[config.COL_TYPE] == 0
                    # Loop through horizons
                    for horizon in range(1, max_horizon + 1):
                        horizon_dt = timedelta(days=horizon)
                        eval_end_time = eval_time + horizon_dt
                        # Slice out the real timeline
                        horizon_mask_ac = (full_adm[config.COL_AGE] > eval_time) & (
                            full_adm[config.COL_AGE] <= eval_end_time
                        )
                        act = full_adm.loc[horizon_mask_ac]  # <- no demographic rows
                        # Slice out the simulation results
                        horizon_mask_sim = (sim_result[config.COL_AGE] > eval_time) & (
                            sim_result[config.COL_AGE] <= eval_end_time
                        )
                        sim = sim_result.loc[horizon_mask_sim | dmg_mask_sim]
                        # Collect stats
                        n_sim = sim[config.COL_PID].nunique()
                        # *** DEBUG ***
                        if n_sim != 256:
                            print("Irregular n-sim encountered")
                            print("n-sim", n_sim)
                            print(sim_result_path)
                        # *************
                        # NOTE: Here, sim_pv contains patient_id (simulation no) and selected_codes in the column
                        #   sim_pv.shape = (n_total_labs, n_selected_codes + 1)
                        #   You can slice lab results from a specific simulation by slicing using patient_id
                        sim_pv = _collect_pivot_tables_dist(
                            sim, selected_codes=selected_codes
                        )
                        # Collect actual pivot table
                        act_pv = _collect_pivot_tables_dist(
                            act, selected_codes=selected_codes
                        )
                        # Add to the dictionary
                        horizon_key = f"{horizon}_days"
                        if horizon_key not in act_pivots:
                            act_pivots[horizon_key] = []
                        if horizon_key not in sim_pivots:
                            sim_pivots[horizon_key] = []
                        act_pivots[horizon_key].append(act_pv)
                        sim_pivots[horizon_key].append(sim_pv)

                except (FileNotFoundError, EOFError) as e:
                    print(f"Failed to open {sim_result_path}")
                    print("ERROR:", e)
                    print("*****TRACEBACK*******")
                    traceback.print_exc()
                    print("*********************")
                    continue

    # Because data can be very large, save data by patient and horizon
    pid = os.getpid()
    random_file_id = uuid4().hex
    actual_temp_dir = os.path.join(temp_dir, "actual")
    for horizon_key, act_pv_list in act_pivots.items():
        # Create a child directory to prevent too many files in the temp directory
        actual_child_dir = os.path.join(actual_temp_dir, horizon_key, str(pid))
        os.makedirs(actual_child_dir, exist_ok=True)
        if len(act_pv_list) > 0:
            temp_file_name = os.path.join(
                actual_child_dir, f"act_{horizon_key}_{random_file_id}.pkl"
            )
            with open(temp_file_name, "wb") as f:
                pickle.dump(act_pv_list, f)
    sim_temp_dir = os.path.join(temp_dir, "sim")
    for horizon_key, sim_pv_list in sim_pivots.items():
        sim_child_dir = os.path.join(sim_temp_dir, horizon_key, str(pid))
        os.makedirs(sim_child_dir, exist_ok=True)
        if len(sim_pv_list) > 0:
            temp_file_name = os.path.join(
                sim_child_dir, f"sim_{horizon_key}_{random_file_id}.pkl"
            )
            with open(temp_file_name, "wb") as f:
                pickle.dump(sim_pv_list, f)


def eval_mc_adm_lab_dist(
    output_dir: str,
    result_dir: str,
    dataset_dir: str,
    max_horizon: int = 7,
    code_percentile: int = 80,
    max_workers: int | None = None,
):
    # Select target codes
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        debug=os.environ.get("DEBUG_MODE") == "1",
        debug_chunks=int(os.environ.get("DEBUG_CHUNKS", "10")),
    )
    settings_manager.write()
    try:
        # Select codes
        num_stats = load_numeric_stats(file_path=None, single_unit=True).sort_values(
            "count", ascending=False
        )
        selected_codes = _select_codes(
            df=num_stats[[config.COL_ITEM_CODE, "count"]], percent=code_percentile
        )
        # selected_codes = list(GENERAL_IVDS.values())
        if max_workers is None:
            max_workers = psutil.cpu_count(logical=False) - 1
        patient_list = pd.read_pickle(os.path.join(result_dir, "patient_list.pkl"))
        if not result_dir.endswith("/"):
            result_dir = result_dir + "/"
        patient_list["path"] = result_dir + patient_list["path"]
        with open(os.path.join(result_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        time_of_eval = metadata["time_of_eval"]
        max_days = metadata["max_days"]
        time_of_eval_td = timedelta(hours=time_of_eval)

        # Debug handling
        if os.environ.get("DEBUG_MODE") == "1":
            debug_chunks = os.environ.get("DEBUG_CHUNKS", str(max_workers * 10))
            debug_chunks = int(debug_chunks)
            patient_list = patient_list.iloc[:debug_chunks]

        with TemporaryDirectory(dir=output_dir) as temp_dir:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _eval_mc_adm_lab_dist,
                        patient_dir=patient_dir,
                        selected_codes=selected_codes,
                        time_of_eval_td=time_of_eval_td,
                        max_horizon=max_horizon,
                        max_days=max_days,
                        temp_dir=temp_dir,
                    )
                    for patient_dir in patient_list["path"]
                ]
                for future in tqdm(
                    as_completed(futures), total=len(patient_list["path"])
                ):
                    _ = future.result()

            # Aggregate results
            print("Aggregating results...")
            output_child_dir = os.path.join(output_dir, "lab_dist_results")
            os.makedirs(output_child_dir, exist_ok=True)
            # NOTE: Exit the pool once, to ensure all child proceses release memory
            for sim_or_act in ["act", "sim"]:
                for horizon in range(1, max_horizon + 1):
                    # Collect all files
                    temp_files = glob.glob(
                        os.path.join(temp_dir, "**", f"{sim_or_act}_{horizon}_*.pkl"),
                        recursive=True,
                    )
                    if len(temp_files) == 0:
                        print(f"No {sim_or_act} files found for horizon {horizon}")
                        continue
                    # Load all files
                    all_pivots = []
                    for temp_file in temp_files:
                        with open(temp_file, "rb") as f:
                            pivots = pickle.load(f)
                        all_pivots.extend(pivots)
                    # Save the aggregated file
                    output_file = os.path.join(
                        output_child_dir, f"lab_dist_{sim_or_act}_{horizon}_days.pkl"
                    )
                    with open(output_file, "wb") as f:
                        pickle.dump(all_pivots, f)
                    # remove temporary files
                    for temp_file in temp_files:
                        os.remove(temp_file)

    finally:
        settings_manager.delete()


# **********
# * Length *
# **********


def _eval_mc_adm_length_calib(
    patient_dir: str,
    time_of_eval_td: timedelta,
    max_horizon: timedelta,
    max_days: int,
    n_bins: int,
):
    # Load the full timelineand apply a custom function if indicated
    patient_no = patient_dir.split("/")[-1]
    total_stats = {}
    full_timeline = pd.read_pickle(os.path.join(patient_dir, "full_timeline.pkl"))
    admissions = [d for d in os.listdir(patient_dir) if d.startswith("admission")]
    for adm in admissions:
        # Load the time of admission and discharge
        with open(os.path.join(patient_dir, adm, "admission_info.pkl"), "rb") as f:
            info_data = pickle.load(f)
        time_adm, time_dsc, timestamp_adm, dept_adm = info_data
        date_adm = timedelta(days=time_adm.days)
        date_dsc = timedelta(days=time_dsc.days)
        n_days = (date_dsc - date_adm).days
        n_days = min(max_days, n_days)
        # Slice full real timeline with decent pad
        # NOTE: At this time, demographics are removed! Be careful!
        full_adm_mask = (full_timeline[config.COL_AGE] > time_adm) & (
            full_timeline[config.COL_AGE] <= (time_dsc + timedelta(days=max_horizon))
        )
        full_adm = full_timeline.loc[full_adm_mask].copy()

        # Read through the simulation results per day
        for day in range(0, n_days + 1):
            eval_time = date_adm + timedelta(days=day) + time_of_eval_td
            if (time_adm <= eval_time) & (eval_time <= time_dsc):
                try:
                    # Open a simulation result
                    sim_result_path = os.path.join(patient_dir, adm, f"day{day}.pkl")
                    sim_result = pd.read_pickle(sim_result_path)
                    # Slice out the simulation results
                    dmg_mask_sim = sim_result[config.COL_TYPE] == 0
                    # Loop through horizons
                    for horizon in range(1, max_horizon + 1):
                        horizon_dt = timedelta(days=horizon)
                        eval_end_time = eval_time + horizon_dt
                        # Slice out the real timeline
                        horizon_mask_ac = (full_adm[config.COL_AGE] > eval_time) & (
                            full_adm[config.COL_AGE] <= eval_end_time
                        )
                        act = full_adm.loc[
                            horizon_mask_ac
                        ]  # <- this already does not contain demographics!
                        # Slice out the simulation results
                        horizon_mask_sim = (sim_result[config.COL_AGE] > eval_time) & (
                            sim_result[config.COL_AGE] <= eval_end_time
                        )
                        sim = sim_result.loc[
                            horizon_mask_sim | dmg_mask_sim
                        ]  # <- You need to preserve demographics to count number of simulations.
                        # Collect stats
                        n_sim = sim[config.COL_PID].nunique()
                        # *** DEBUG ***
                        if n_sim != 256:
                            print("Irregular n-sim encountered")
                            print("n-sim", n_sim)
                            print(sim_result_path)
                        # *************
                        # Compute upper and lower bounds of estimated length percents
                        lower_bounds, upper_bounds = _compute_length_calibration(
                            sim, n_bins=n_bins
                        )
                        # Compute actual length
                        actual_length = _count_tokens(act)
                        # Compute rates the actual length is within the bounds
                        within_lower = actual_length >= lower_bounds
                        within_upper = actual_length <= upper_bounds
                        within_range = within_lower & within_upper
                        # Finalize
                        within_range = within_range.astype(
                            int
                        )  # <- 0 or 1, shape is (n_bins,)
                        # Add to the dictionary
                        horizon_key = f"{horizon}_days"
                        if horizon_key not in total_stats:
                            total_stats[horizon_key] = []
                        total_stats[horizon_key].append(within_range)

                except (FileNotFoundError, EOFError) as e:
                    print(f"Failed to open {sim_result_path}")
                    print("ERROR:", e)
                    print("*****TRACEBACK*******")
                    traceback.print_exc()
                    print("*********************")
                    continue

    # Finalize the stats
    for k in total_stats.keys():
        if total_stats[k]:
            total_stats[k] = np.stack(total_stats[k])  # (n_days, n_bins)
        else:
            total_stats[k] = None

    return total_stats


def eval_mc_adm_length_calib(
    result_dir: str,
    dataset_dir: str,
    n_bins: str = 21,  # '21' for 20 equal-width bins, like 0, 5, 10, ..., 100.
    max_horizon: int = 7,
    max_workers: int | None = None,
) -> dict:
    # Select target codes
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        debug=os.environ.get("DEBUG_MODE") == "1",
        debug_chunks=int(os.environ.get("DEBUG_CHUNKS", "10")),
    )
    settings_manager.write()
    try:
        if max_workers is None:
            max_workers = psutil.cpu_count(logical=False) - 1
        patient_list = pd.read_pickle(os.path.join(result_dir, "patient_list.pkl"))
        if not result_dir.endswith("/"):
            result_dir = result_dir + "/"
        patient_list["path"] = result_dir + patient_list["path"]
        with open(os.path.join(result_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        time_of_eval = metadata["time_of_eval"]
        max_days = metadata["max_days"]
        time_of_eval_td = timedelta(hours=time_of_eval)

        # Debug handling
        if os.environ.get("DEBUG_MODE") == "1":
            debug_chunks = os.environ.get("DEBUG_CHUNKS", str(max_workers * 10))
            debug_chunks = int(debug_chunks)
            patient_list = patient_list.iloc[:debug_chunks]

        result = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _eval_mc_adm_length_calib,
                    patient_dir=patient_dir,
                    n_bins=n_bins,
                    time_of_eval_td=time_of_eval_td,
                    max_horizon=max_horizon,
                    max_days=max_days,
                )
                for patient_dir in patient_list["path"]
            ]
            for future in tqdm(as_completed(futures), total=len(patient_list["path"])):
                sub_result = future.result()
                for k, v in sub_result.items():
                    if k not in result:
                        result[k] = []
                    if v is not None:
                        result[k].append(v)

        # Finalize the result
        for k in result.keys():
            result[k] = np.concatenate(result[k], axis=0)  # (n_total_days, n_bins)

        return result

    finally:
        settings_manager.delete()


# *************************
# * Baseline acquisitions *
# *************************
def _create_sampling_table_fn(
    labelled_path: str,
    hour_of_sampling: int | None = None,
    max_days: int = None,
    min_days: int = 0,
    train_only: bool = True,
    period_start: str | None = None,
    period_end: str | None = None,
) -> pd.DataFrame:
    """Create sampling weights for sampling training data to generate baseline.
    Because data are saved in multiple files, this is needed.

    Args:
        labelled_path (str): Path to the labelled data file.
        hour_of_sampling (int|None): Hour of sampling. If None, nothing is excluded.
        max_days (int): Maximum number of days from an admission.
        min_days (int): Minimum number of days from an admission.
        train_only (bool): If True, only training data is considered.
        period_start (str|None): Start of the period to consider in %Y/%m/%d format. If None, no start is applied.
        period_end (str|None): End of the period to consider in %Y/%m/%d format. If None, no end is applied.
    """
    # Load
    df = pd.read_pickle(labelled_path)
    df = df[
        [
            "patient_id",
            "timestamp",
            "timedelta",
            "admitted",
            "original_value",
            "train_period",
        ]
    ]
    # Select valid rows
    if train_only:
        df = df.loc[df["train_period"] == 1]
    if period_start is not None:
        period_start = pd.to_datetime(period_start, format="%Y/%m/%d")
        df = df.loc[df["timestamp"] >= period_start]
    if period_end is not None:
        period_end = pd.to_datetime(period_end, format="%Y/%m/%d")
        df = df.loc[df["timestamp"] <= period_end]
    # Select only rows with admission and discharge
    time_of_adm = df.loc[
        df["original_value"] == "[ADM]", ["patient_id", "timedelta"]
    ].copy()
    time_of_adm = time_of_adm.loc[time_of_adm["timedelta"].notna()]
    time_of_dsc = df.loc[
        df["original_value"] == "[DSC]", ["patient_id", "timedelta"]
    ].copy()
    time_of_dsc = time_of_dsc.loc[time_of_dsc["timedelta"].notna()]
    # Concatenate admission and discharge times
    time_of_adm = time_of_adm.rename(columns={"timedelta": "time_adm"})
    time_of_dsc = time_of_dsc.rename(columns={"timedelta": "time_dsc"})
    days_df = pd.merge(
        time_of_adm,
        time_of_dsc,
        on="patient_id",
        how="left",
    )  # <- This results in many duplicates! you need to drop them later
    # Select valud dsc time only
    days_df["diff"] = days_df["time_dsc"] - days_df["time_adm"]
    days_df = days_df.loc[
        days_df["diff"] >= timedelta(seconds=0)
    ]  # Exclude discharge times before admission
    days_df = days_df.sort_values("diff", ascending=True).drop_duplicates(
        subset=["patient_id", "time_adm"]
    )  # Drop duplicates
    # Determine the start of counts
    days_df["start"] = days_df["time_adm"].dt.floor("D")
    days_df["end"] = days_df["time_dsc"].dt.floor("D")
    # Prevent the final day of admission from being counted if the discharge is before the hour of sampling
    if hour_of_sampling is not None:
        dsc_before_sampling_hour = (
            days_df["time_dsc"].dt.total_seconds() / 3600 < hour_of_sampling
        )
        # Make the end day -1
        days_df.loc[dsc_before_sampling_hour, "end"] -= timedelta(days=1)
    # Min days handlings
    if min_days == 0:
        # Prevent the first day of admission from being counted if the admission is after the hour of sampling
        if hour_of_sampling is not None:
            adm_after_sampling_hour = (
                days_df["time_adm"].dt.total_seconds() / 3600 > hour_of_sampling
            )
            # Make the initial day +1
            days_df.loc[adm_after_sampling_hour, "start"] += timedelta(days=1)
    else:
        days_df["start"] += timedelta(days=min_days)
    # Max days handling
    if max_days is not None:
        days_df["end"] = np.minimum(
            days_df["end"], days_df["start"] + timedelta(days=max_days - 1)
        )
    # Check consistency of start and end dates
    negative_days_mask = days_df["end"] < days_df["start"]
    if negative_days_mask.any():
        days_df = days_df.loc[~negative_days_mask]
    # Generate list of sampling dates for each row
    base_date = pd.Timestamp("2000-01-01")  # Dummy date for computing date range
    days_df["start_dt"] = base_date + days_df["start"]
    days_df["end_dt"] = base_date + days_df["end"]
    days_df["sampling_dates_dt"] = days_df.apply(
        lambda row: pd.date_range(start=row["start_dt"], end=row["end_dt"], freq="D"),
        axis=1,
    )
    # Subtract base_date from each date in the range
    days_df["sampling_dates"] = days_df["sampling_dates_dt"].apply(
        lambda arr: arr - base_date
    )
    # Explode to have one row per (patient_id, sampling_date)
    days_df = days_df.explode("sampling_dates").reset_index(drop=True)
    # Add source file name
    days_df["source_file"] = labelled_path
    # Organize columns
    days_df = days_df[["patient_id", "sampling_dates", "source_file"]]

    return days_df


def create_baseline_sampling_table(
    tables_dir: str,
    max_workers: int,
    saved_dir: str,
    hour_of_sampling: int | None = None,
    max_days: int | None = None,
    min_days: int = 0,
    train_only: bool = True,
    period_start: str | None = None,
    period_end: str | None = None,
):
    """Create sampling weights for sampling training data to generate baseline."""
    # Load
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _create_sampling_table_fn,
                labelled_path=os.path.join(tables_dir, f),
                hour_of_sampling=hour_of_sampling,
                max_days=max_days,
                min_days=min_days,
                train_only=train_only,
                period_start=period_start,
                period_end=period_end,
            )
            for f in os.listdir(tables_dir)
            if "train" in f
        ]
        total_count_df = None
        # Collect df
        for future in tqdm(as_completed(futures), total=len(futures)):
            count_df = future.result()
            if count_df.size:
                if total_count_df is None:
                    total_count_df = count_df
                else:
                    total_count_df = pd.concat(
                        [total_count_df, count_df], ignore_index=True
                    ).reset_index(drop=True)
        # Finalize df
        total_count_df = total_count_df.sort_values(
            ["patient_id", "sampling_dates"]
        ).reset_index(drop=True)

        # Save
        saved_path = os.path.join(saved_dir, "baseline_sampling_table.pkl")
        os.makedirs(saved_dir, exist_ok=True)
        total_count_df.to_pickle(saved_path)


def draw_baseline_samples(
    sampling_table_path: str,
    n_samples: int = None,
    replace: bool = False,
) -> pd.DataFrame:
    """Draw baseline samples from the sampling weights.

    Args:
        sampling_table_path (str): Path to the sampling table.
        n_samples (int|None): Number of samples to draw. If None, all samples are drawn.
        replace (bool): Whether to sample with replacement. Default is False.
    Returns:
        pd.DataFrame: Sampled dataframe with columns ['patient_id', 'sampling_dates', 'source_file'].
    """
    # Load
    sampling_table_df = pd.read_pickle(sampling_table_path)
    if n_samples is None:
        n_samples = sampling_table_df.shape[0]
    # Draw samples
    if n_samples > sampling_table_df.shape[0]:
        raise ValueError(
            f"Requested {n_samples} samples, but only {sampling_table_df.shape[0]} available."
        )
    sampled_df = sampling_table_df.sample(n=n_samples, replace=replace)
    # Reset index
    sampled_df = sampled_df.reset_index(drop=True)
    # Return
    return sampled_df


def count_available_baseline_samples(
    sampling_table_path: str,
) -> int:
    """Count available baseline samples from the sampling weights."""
    # Load
    sampling_table_df = pd.read_pickle(sampling_table_path)
    # Count
    n_samples = sampling_table_df.shape[0]
    return n_samples


def baseline_day_sampler(
    sampling_table_path: str,
    n_samples: int,
):
    """Draw baseline samples from the sampling weights."""
    sampled_df = draw_baseline_samples(
        sampling_table_path=sampling_table_path,
        n_samples=n_samples,
    )
    files = sampled_df["source_file"].unique()
    # Load by files
    for file in files:
        # Slice the sampled dataframe for the current file
        file_df = sampled_df.loc[sampled_df["source_file"] == file]
        yield file, file_df


def _get_baseline_distribution_helper(
    file: str,
    file_df: pd.DataFrame,
    fn: Callable,
    temp_dir: str,
    min_horizon: int = 1,
    max_horizon: int = 7,
    hour_of_sampling: int = None,
    **kwargs,
) -> dict:
    """Helper function to get baseline distribution."""
    # Load lablled data tabel
    labelled_df = pd.read_pickle(file)
    # Iterate over horizons (May not be efficient, but for memory footprint and analysis, saving results by horizon is effecitve overall.)
    for horizon in range(min_horizon, max_horizon + 1):
        # Initialize a list to hold day-level stats
        horizon_level_stats = []
        # Check by patients
        for patient_id, patient_days_df in file_df.groupby("patient_id"):
            # Get the days
            days = patient_days_df["sampling_dates"].values
            # Slice patient data only
            patient_data = labelled_df.loc[labelled_df["patient_id"] == patient_id]
            for hosp_day in days:
                hosp_day = pd.to_timedelta(hosp_day)
                # Start of sampling time
                if hour_of_sampling is not None:
                    start = hosp_day + timedelta(hours=hour_of_sampling)
                else:
                    # Get the nearest admission time
                    adm_mask = patient_data["original_value"] == "[ADM]"
                    adm_times = patient_data.loc[adm_mask, "timedelta"]
                    adm_times = adm_times[adm_times < hosp_day]
                    adm_times = adm_times.sort_values(ascending=False)
                    # Ensure that starging time is after the admission time
                    if not adm_times.empty:
                        adm_time = adm_times.iloc[0]
                        sample_range = min(
                            hosp_day.total_seconds() - adm_time.total_seconds(),
                            24 * 3600,
                        )
                        time_of_sampling = np.random.uniform(0, sample_range)
                        start = adm_time + timedelta(seconds=time_of_sampling)
                    else:
                        print(
                            "WARNING: No admission time found for patient", patient_id
                        )
                        continue  # No admission before this day, which is not expected.
                # End of sampling time
                end = start + timedelta(days=horizon)
                # Get the data for the patient
                horizon_data = patient_data.loc[
                    (patient_data["timedelta"] >= start)
                    & (patient_data["timedelta"] <= end)
                ]
                # Get the distribution
                stats = fn(horizon_data, **kwargs)
                # Append to the list
                horizon_level_stats.append(stats)

        # Save data by horizons
        if horizon_level_stats:
            saved_path = os.path.join(
                temp_dir, f"{horizon}_days_baseline_distribution_{uuid4().hex}.pkl"
            )
            with open(saved_path, "wb") as f:
                pickle.dump(horizon_level_stats, f)

    return


def get_baseline_distribution(
    sampling_table_path: str,
    n_samples: int | None,
    max_workers: int,
    fn: Callable,
    output_dir: str,
    min_horizon: int = 1,
    max_horizon: int = 7,
    hour_of_sampling: int = None,
    **kwargs: dict,
) -> dict:
    """Get baseline distribution of sampled data."""
    # Sampler
    sampler = baseline_day_sampler(
        sampling_table_path=sampling_table_path,
        n_samples=n_samples,
    )
    # Multiprocess
    with TemporaryDirectory(dir=output_dir) as temp_dir:
        # Collect stats and save by horizons (chunked)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _get_baseline_distribution_helper,
                    file=file,
                    file_df=file_df,
                    fn=fn,
                    temp_dir=temp_dir,
                    min_horizon=min_horizon,
                    max_horizon=max_horizon,
                    hour_of_sampling=hour_of_sampling,
                    **kwargs,
                )
                for file, file_df in sampler
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                _ = future.result()

        # Aggregate chunked files
        for horizon in range(min_horizon, max_horizon + 1):
            horizon_key = f"{horizon}_days"
            horizon_files = [
                os.path.join(temp_dir, tempfname)
                for tempfname in os.listdir(temp_dir)
                if horizon_key in tempfname
            ]
            horizon_level_stats = []
            for horizon_file in horizon_files:
                with open(horizon_file, "rb") as f:
                    horizon_data = pickle.load(f)
                    horizon_level_stats.extend(horizon_data)
            # Save aggregated data
            if horizon_level_stats:
                saved_path = os.path.join(
                    output_dir, f"{horizon_key}_baseline_distribution.pkl"
                )
                with open(saved_path, "wb") as f:
                    pickle.dump(horizon_level_stats, f)
