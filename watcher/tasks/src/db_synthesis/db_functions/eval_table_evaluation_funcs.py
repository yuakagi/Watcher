"""Helpful functions to evaluate synthetic tables."""

import os
import glob
import random
from typing import Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
from .who_general_ivds import GENERAL_IVD
from .....general_params import watcher_config as config
from .....general_params import BaseSettingsManager, get_settings, define_max_workers
from .....utils import (
    tally_stats,
    extract_numeric_from_text,
    load_patient_id_dict,
)


def _select_by_period(df: pd.DataFrame, flag: str = None) -> pd.DataFrame:
    if flag is not None:
        cols = df.columns
        if not ((config.COL_TRAIN_PERIOD in cols) and (config.COL_TEST_PERIOD in cols)):
            raise KeyError(
                f"{config.COL_TRAIN_PERIOD} and {config.COL_TEST_PERIOD} must be in the dataframe."
            )
        dmg_rows = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DMG]
        if flag in [config.TRAIN, config.VAL]:
            period_mask = df[config.COL_TRAIN_PERIOD] == 1
            df = df.loc[period_mask | dmg_rows, :]
        elif flag == config.TRAIN:
            period_mask = df[config.COL_TEST_PERIOD] == 1
            df = df.loc[period_mask | dmg_rows, :]
        else:
            raise ValueError(f"Unexpected value ({flag}) passed to 'flag' arg.")
    return df


def _map_eval_tables_wrapper(file: str, custom_fn: Callable, id_dict: dict, **kwargs):
    """Wrapper function for parallel_map_eval_tables.
    Reads a table file, and applies the custom function to it.
    """
    # Determine train-test-val flag
    parts = file.split("_")
    flag = parts[-2]
    if flag not in [config.TRAIN, config.VAL, config.TEST]:
        flag = None
    # Read table
    df = pd.read_pickle(file)
    df = _select_by_period(df, flag)
    if id_dict is not None:
        ids = id_dict[flag]
        df = df.loc[df[config.COL_PID].isin(ids)]
    # Apply function
    result = custom_fn(df, **kwargs)
    return result


def paralell_map_eval_tables(
    file_pattern: str,
    custom_fn: Callable,
    max_workers: int = None,
    id_dict: dict = None,
    **kwargs,
) -> list[Any]:
    """Applies a custom function to synthetic tables using multiprocessing."""
    if "**" in file_pattern:
        files = glob.glob(file_pattern, recursive=True)
    else:
        files = glob.glob(file_pattern)
    if max_workers is None:
        max_workers = define_max_workers()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = []
        futures = [
            executor.submit(
                _map_eval_tables_wrapper,
                file=file,
                custom_fn=custom_fn,
                id_dict=id_dict,
                **kwargs,
            )
            for file in files
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Working on synthetic tables",
        ):
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def flag_admission_status_for_eval(
    df: pd.DataFrame, ignore_truncated: bool = False
) -> pd.Series:
    """Puts binary admission status flags (0 or 1) to the records in evaluation tables.
    Args:
        df (pd.DataFrame): Evaluation table.
        ignore_truncated (bool): If true, truncated admission histories (missing either [ADM] or [DSC])
            are not counted as admitted.
            In case you want to collect statistics on admissions and discharges (such as length of stay),
            consider setting this argument to false.
    Returns:
        binary_admission_status (pd.Series): Series of binary flags for admission status.
            Records within an admission is flagged with 1.
    """
    # Add a column for admission status
    dmg_mask = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DMG]
    is_adm = df[config.COL_CODE] == "[ADM]"
    is_dsc = df[config.COL_CODE] == "[DSC]"
    adm_based = is_adm.astype(float)
    dsc_based = is_dsc.astype(float)
    adm_based.loc[(~is_adm)] = None
    dsc_based.loc[(~is_dsc)] = None
    adm_based.loc[dmg_mask | is_dsc] = 0.0
    # TODO (Yu Akagi): DSC may be counted as adm, as length of staty is [ADM] ~ [DSC]. Not mandatory.
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


def _collect_general_stats_from_eval_tables(df: pd.DataFrame) -> dict:
    """Collect general statistics from evaluation tables."""
    # Total number of patients
    n_patients = int(df[config.COL_PID].nunique())
    # Count genders
    n_men = int((df[config.COL_CODE] == "[M]").sum())
    n_women = int((df[config.COL_CODE] == "[F]").sum())
    # Count ages
    init_age_mask = df[config.COL_PID] != df[config.COL_PID].shift(1, fill_value="")
    init_ages = df.loc[init_age_mask, config.COL_AGE].dt.days // 365
    init_ages = init_ages.tolist()
    # TODO (YuAkagi):It is hard to compute the timeline lengths of synthetic data. Refine it if necessary.
    # Count periods
    end_age_mask = init_age_mask.shift(-1)
    end_age_mask.iloc[-1] = True
    init_timedelta = df.loc[init_age_mask, config.COL_AGE]
    end_timedelta = df.loc[end_age_mask, config.COL_AGE]
    periods = (
        end_timedelta.dt.total_seconds().values
        - init_timedelta.dt.total_seconds().values
    )
    periods = np.ceil(periods / 60 / 60 / 24).tolist()
    # Collect stats related to admissions
    # NOTE: Admission is counted in a way that truncated admissions are ignored.
    #       This is sigihtly different from COL_ADM.
    adm_status = flag_admission_status_for_eval(df, ignore_truncated=True)
    shifted_adm_status = adm_status.shift(1, fill_value=0)
    adm_starts = (adm_status == 1) & (shifted_adm_status == 0)
    adm_ends = (adm_status == 0) & (shifted_adm_status == 1)
    n_nontruncated_admissions = int(adm_starts.sum())
    length_of_stays = (
        df.loc[adm_ends, config.COL_AGE].dt.total_seconds().values
        - df.loc[adm_starts, config.COL_AGE].dt.total_seconds().values
    )
    length_of_stays = np.ceil(length_of_stays / 60 / 60 / 24).tolist()
    dispositions = df.loc[adm_ends, config.COL_RESULT]
    n_survived = (dispositions == "[DSC_ALV]").sum()
    n_expired = (dispositions == "[DSC_EXP]").sum()
    n_others = (dispositions == "[DSC_OTHER]").sum()
    # Count major diagnoses
    dx_type_mask = df[config.COL_TYPE] == config.RECORD_TYPE_NUMBERS[config.DX]
    prov_mask = df[config.COL_CODE].str.endswith(config.PROV_SUFFIX, na=False)
    non_prov = ~prov_mask
    dx_mask = dx_type_mask & non_prov
    r_icd10_cancer = r"(^C[0-8][0-9])|(^C9[0-7])|(^D0[0-9])"
    r_icd10_ht = r"^I1[0-5]"
    r_icd10_dm = r"^E1[0-4]"
    r_icd10_dl = r"^E78"
    cancer_mask = df[config.COL_CODE].str.match(r_icd10_cancer, na=False)
    ht_mask = df[config.COL_CODE].str.match(r_icd10_ht, na=False)
    dm_mask = df[config.COL_CODE].str.match(r_icd10_dm, na=False)
    dl_mask = df[config.COL_CODE].str.match(r_icd10_dl, na=False)
    n_cancer_patients = int(df.loc[cancer_mask & dx_mask, config.COL_PID].nunique())
    n_ht_patients = int(df.loc[ht_mask & dx_mask, config.COL_PID].nunique())
    n_dm_patients = int(df.loc[dm_mask & dx_mask, config.COL_PID].nunique())
    n_dl_patients = int(df.loc[dl_mask & dx_mask, config.COL_PID].nunique())

    # Organize the stats
    stats = {
        "n_patients": n_patients,
        "sex": {"number_of_men": n_men, "number_of_women": n_women},
        "age": {"ages": init_ages},
        "period": {"periods": periods},
        "admission": {
            "number_of_admissions": n_nontruncated_admissions,
            "number of discharges alived": n_survived,
            "number_of_discharges_expired": n_expired,
            "number of discharges others": n_others,
            "lengths_of_stay": length_of_stays,
        },
        "major_diagnoses": {
            "patients with cancer": n_cancer_patients,
            "patients with hypertension": n_ht_patients,
            "patients with diabetes mellitus": n_dm_patients,
            "patients wiht dyslipidemia": n_dl_patients,
        },
    }
    return stats


def collect_general_stats_from_eval_tables(
    file_pattern: str, max_workers: int, id_dict: dict = None
) -> dict:
    """Collects general statistics from evaluation tables."""
    list_of_stats = paralell_map_eval_tables(
        file_pattern=file_pattern,
        custom_fn=_collect_general_stats_from_eval_tables,
        max_workers=max_workers,
        id_dict=id_dict,
    )
    # Clean the nested stats
    final_stats = tally_stats(list_of_stats)
    male_female_ratio = (
        final_stats["sex"]["number_of_men"] / final_stats["sex"]["number_of_women"]
    )
    ages = np.array(final_stats[config.COL_AGE]["ages"])
    periods = np.array(final_stats["period"]["periods"])
    n_admissions = final_stats["admission"]["number_of_admissions"]
    inhospital_death_rate = np.round(
        final_stats["admission"]["number_of_discharges_expired"] / n_admissions, 4
    )
    los = np.array(final_stats["admission"]["lengths_of_stay"])

    cleaned_stats = {
        "n_patients": final_stats["n_patients"],
        "sex": {"male_to_female": male_female_ratio},
        "age": {
            "mean": ages.mean(),
            "median": np.median(ages),
            "std": ages.std(),
            "min": ages.min(),
            "max": ages.max(),
        },
        "period": {
            "mean": periods.mean(),
            "median": np.median(periods),
            "std": periods.std(),
            "min": periods.min(),
            "max": periods.max(),
        },
        "admission": {
            "n_admissions": n_admissions,
            "inhospital_death_rate": inhospital_death_rate,
            "length_of_stay": {
                "mean": los.mean(),
                "median": np.median(los),
                "std": los.std(),
                "min": los.min(),
                "max": los.max(),
            },
        },
        "major_diagnoses": final_stats["major_diagnoses"],
    }

    return cleaned_stats


def _count_codes_in_eval_tables(df: pd.DataFrame) -> dict:
    """Counts codes seen per database and per patient."""
    # Record types
    dx_types = [config.RECORD_TYPE_NUMBERS[config.DX]]
    pharma_types = [
        config.RECORD_TYPE_NUMBERS[config.PSC_O],
        config.RECORD_TYPE_NUMBERS[config.INJ_O],
    ]
    lab_types = [config.RECORD_TYPE_NUMBERS[config.LAB_R]]

    # Initialize dict
    count_dict = {"per_db": {}, "per_patient": {}}

    # Counting per database
    total_patients = df[config.COL_PID].nunique()
    total_codes = 0
    for key, t in zip(
        [config.DX_CODE, config.MED_CODE, config.LAB_CODE],
        [dx_types, pharma_types, lab_types],
    ):
        mask = df[config.COL_TYPE].isin(t)
        masked_df = df.loc[mask, :]
        # Counting per database
        n_codes = len(masked_df[config.COL_CODE])
        total_codes += n_codes
        counts_per_db = masked_df[config.COL_CODE].value_counts().to_dict()
        # Counting per patient
        dup_removed = masked_df.drop_duplicates(
            subset=[config.COL_PID, config.COL_CODE]
        )
        counts_per_patient = dup_removed[config.COL_CODE].value_counts().to_dict()
        # Save the results
        count_dict["per_db"][key] = {"n_codes": n_codes, "value_counts": counts_per_db}
        count_dict["per_patient"][key] = {"value_counts": counts_per_patient}
    count_dict["per_db"]["total"] = total_codes
    count_dict["per_patient"]["total"] = total_patients
    return count_dict


def count_codes_in_eval_tables(
    file_pattern: str,
    max_workers: int = None,
    id_dict: dict = None,
) -> pd.DataFrame:
    """Compute code prevalence in the synthetic data."""
    list_of_stats = paralell_map_eval_tables(
        file_pattern=file_pattern,
        custom_fn=_count_codes_in_eval_tables,
        max_workers=max_workers,
        id_dict=id_dict,
    )
    final_stats = tally_stats(config.MED_CODE)
    for k in [config.DX_CODE, config.MED_CODE, config.LAB_CODE]:
        final_stats["per_db"][k]["value_counts"] = [
            final_stats["per_db"][k]["value_counts"]
        ]
        final_stats["per_patient"][k]["value_counts"] = [
            final_stats["per_patient"][k]["value_counts"]
        ]

    return final_stats


def _collect_pivot_tables_from_eval_tables(df, selected_codes):
    """Create a pivot table from a lab table."""
    # Select records
    df = df.loc[df[config.COL_CODE].isin(selected_codes), :].copy()
    # Pivot table
    df[config.COL_RESULT] = extract_numeric_from_text(df[config.COL_RESULT])
    df = df.loc[~df[config.COL_RESULT].isna(), :]
    index_cols = [config.COL_PID, config.COL_AGE]
    df = df.drop_duplicates(subset=[config.COL_PID, config.COL_AGE, config.COL_CODE])
    pv = df.pivot(index=index_cols, columns=config.COL_CODE, values=config.COL_RESULT)
    pv.columns = pv.columns.tolist()
    pv = pv.reset_index(level=0, drop=True)
    pv = pv.astype(float)
    # Ensure the table has all the columns
    for col in selected_codes:
        if col not in pv.columns:
            pv[col] = None
    pv = pv[selected_codes]

    return pv


def correlation_matrix_from_eval_tables(
    file_pattern: str,
    selected_codes: list[str] = None,
    max_workers: int = None,
    id_dict: dict = None,
) -> dict:
    """Creates a correlation matrix of laboratory values.
    This function loads the 'cleaned' laboratory result tables.
    """
    # TODO(Yu Akagi): Needs to select patient IDs when using real data
    if selected_codes is None:
        selected_codes = list(GENERAL_IVD.values())

    pivot_tables = paralell_map_eval_tables(
        file_pattern=file_pattern,
        custom_fn=_collect_pivot_tables_from_eval_tables,
        max_workers=max_workers,
        id_dict=id_dict,
        selected_codes=selected_codes,
    )
    total_pitvot = pd.concat(pivot_tables)
    corr_matrix = total_pitvot.corr(method="pearson", numeric_only=True, min_periods=10)
    return corr_matrix


def _patient_selection_wrapper(
    file: str,
    flag: str,
    all_ids: list[str],
    max_age: int,
    min_age: int,
) -> dict:
    """Wrapper function for patient selection from the labelled tables."""
    # Read file
    valid_ids = all_ids[flag]
    table = pd.read_pickle(file)
    table = table.loc[table[config.COL_PID].isin(valid_ids)]
    table = table.sort_values([config.COL_PID, config.COL_AGE], ascending=True)
    table = table.drop_duplicates(config.COL_PID, keep="first")
    ages = table[config.COL_AGE].dt.days // 365
    # Selection
    if max_age is not None:
        table = table.loc[ages <= max_age]
    if min_age is not None:
        table = table.loc[ages >= min_age]
    # Create a list of IDs
    selected_ids = table[config.COL_PID].tolist()
    id_dict = {flag: selected_ids}
    return id_dict


def select_patient_ids_from_eval_tables(
    dataset_dir: str,
    train: bool = True,
    max_age: int = None,
    min_age: int = None,
    included_only: bool = True,
    max_workers: int = None,
) -> dict:
    """Select patients that match the criteria from the dataset."""
    # Define environment variables
    if max_workers is None:
        max_workers = define_max_workers()
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        max_workers=max_workers,
    )
    settings_manager.write()
    if train:
        flag = config.TRAIN
    else:
        flag = config.TEST
    real_eval_dir = get_settings("EVAL_TABLES_DIR")
    # Collect files
    file_pattern = os.path.join(
        real_eval_dir, config.EVAL_TABLE_FILE_PATTERN.replace("*", f"{flag}_*")
    )
    files = glob.glob(file_pattern)
    # Get patient ID candidates
    all_ids = load_patient_id_dict(included_only=included_only)
    print("Selecting patients...")
    id_dict_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _patient_selection_wrapper,
                file=file,
                flag=flag,
                all_ids=all_ids,
                max_age=max_age,
                min_age=min_age,
            )
            for file in files
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Selecting patients"
        ):
            result = future.result()
            if result is not None:
                id_dict_list.append(result)
    id_dict = tally_stats(id_dict_list)

    return id_dict


def _bootstrap_func_wrapper(
    func: Callable,
    files: list[str],
    ids: pd.DataFrame,
    iterations: int,
    resample_size: int,
    disable_pbar: bool,
    **kwargs,
) -> list[list[any]]:
    """Helper function for bootstrapping
    This function first resamples patient IDs, then loads all the file given and resample them.
    Finally, it applies the given custom function to each resampled records.
    Args:
        func (Callable): Custom function to be applied to resampled tables.
        files (list): List of paths to tables to be loaded.
        ids (pd.DataFrame): Table of patient IDs.
        train (bool): If true, records used for training are selected. Otherwise, data from test are
            loded.
        resample_size (int): Size of resampling.
        disable_pbar (bool): If true, the progress bar is hidden.
    Returns:
        results (list): List of results. This is a list of lists, and each sublist contains the full results of one bootstrap.
            Each element of a sublist is a result yeilded from a chunk. You need to aggregate these elements.
    """
    # Shuffle files to make it less likely that child processes open a file concurrently
    random.shuffle(files)
    # Resample patient IDs
    resampled_id_tables = []
    for _ in range(iterations):
        resampled_ids = ids.sample(n=resample_size, replace=True)
        resampled_ids["n_sampled"] = (
            resampled_ids.groupby([config.COL_PID]).cumcount().add(1)
        )  # <- Assiging number of resampling to each ID
        resampled_ids["n_sampled"] = resampled_ids["n_sampled"].astype(str)
        resampled_id_tables.append(resampled_ids)

    # Start bootstrapping
    results = []
    with tqdm(files, total=len(files), disable=disable_pbar) as pbar:
        for file in pbar:
            parts = file.split("_")
            flag = parts[-3]
            # Load table
            raw_table = pd.read_pickle(file)
            raw_table = _select_by_period(raw_table, flag)
            if os.environ.get("DEBUG_MODE") == "1":
                raw_table = raw_table[:1e4]
            raw_table = raw_table.reset_index(drop=True)
            raw_table["row_no"] = raw_table.index

            # Repeat resampling <- This is the main loop.
            pbar.set_description(f"bootstrap no 0/{iterations}")
            for i, resampled_ids in enumerate(resampled_id_tables):
                # Select patient IDs
                selected_id_table = resampled_ids.loc[:, [config.COL_PID, "n_sampled"]]
                # Select records
                resampled_table = pd.merge(
                    raw_table, selected_id_table, on=config.COL_PID, how="inner"
                )
                if resampled_table.size:
                    # Assign new patient IDs
                    resampled_table[config.COL_PID] = (
                        "btstrp"
                        + resampled_table["n_sampled"]
                        + resampled_table[config.COL_PID]
                    )
                    # Sort the resampled_table
                    resampled_table = resampled_table.sort_values(
                        by=[config.COL_PID, "row_no"], ascending=[True, True]
                    )
                    resampled_table = resampled_table.drop(
                        ["n_sampled", "row_no"], axis=1
                    )

                    # Apply the custom function, and collect the results
                    file_level_result = func(resampled_table, **kwargs)
                else:
                    file_level_result = None
                if len(results) == i:
                    if file_level_result is None:
                        results.append([])
                    else:
                        results.append([file_level_result])
                else:
                    if file_level_result is not None:
                        results[i].append(file_level_result)

                # Update pbar
                pbar.set_description(f"bootstrap no {i+1}/{iterations}")

    return results


def bootstrap_eval_tables(
    dataset_dir: str,
    iterations: int,
    resample_size: int,
    func: Callable,
    train: bool = True,
    id_dict: dict = None,
    max_workers: int = None,
    **kwargs,
) -> list[list[Any]]:
    """Collect statistics from the dataset by bootstrapping.
    Args:
        selection_criteria (Callable): Criteria to select patients.
            If a criteria is specified, this function first read through all the labelled tables,
            and find patients that match,
            The criteria must be a function that takes a pd.Dataframe (a labelled table) as its first
            argument and nothing else, and it must return a series of patient IDs (list, np.ndarray, or pd.Seires).
            It should look like this:
                def example_criteria(table:pd.DataFrame)->list[str]:
                    ...

    """
    # Define variables
    if max_workers is None:
        max_workers = define_max_workers()
    settings_manager = BaseSettingsManager(
        dataset_dir=dataset_dir,
        max_workers=max_workers,
    )
    settings_manager.write()
    if train:
        flag = config.TRAIN
    else:
        flag = config.TEST
    real_eval_dir = get_settings("EVAL_TABLES_DIR")
    # Collect files
    file_pattern = os.path.join(
        real_eval_dir, config.EVAL_TABLE_FILE_PATTERN.replace("*", f"{flag}_*")
    )
    files = glob.glob(file_pattern)

    # If patient IDs are not explicitly given, all the included patient IDs are used.
    # Select patient IDs if indicated
    if id_dict is None:
        id_dict = load_patient_id_dict(included_only=True)
    # Create a table of patient IDs for resampling (This table has patient ID)
    dfs = []
    for k, v in id_dict.items():
        df = pd.DataFrame({config.COL_PID: v[flag]})
        dfs.append(df)
    id_table = pd.concat(dfs)

    # ***********************
    # * Start bootstrapping *
    # ***********************
    # Determine the number of iterations performed by each child process
    if iterations < max_workers:
        child_iterations = np.ones(iterations)
    else:
        child_iterations = np.full(max_workers, iterations // max_workers)
        remainder = iterations % max_workers
        if remainder:
            child_iterations[:remainder] += 1
    child_iterations = child_iterations.astype(int)
    # Start multiprocessing
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _bootstrap_func_wrapper,
                func=func,
                files=files,
                ids=id_table,
                iterations=k,
                resample_size=resample_size,
                disable_pbar=not i == 0,
                **kwargs,
            )
            for i, k in enumerate(child_iterations)
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Bootstrapping"
        ):
            result = future.result()
            results += result  # <- Do not use .append() here to avoid nesting.

    return results


def bootstrap_general_stats_from_eval_tables(
    dataset_dir: str,
    iterations: int,
    resample_size: int,
    train: bool = True,
    id_dict: dict = None,
    max_workers: int = None,
) -> dict:
    """Samples general statistics from the labelled tables by bootstrapping."""

    # Bootstrap
    nested_stats = bootstrap_eval_tables(
        dataset_dir=dataset_dir,
        iterations=iterations,
        resample_size=resample_size,
        func=_collect_general_stats_from_eval_tables,
        train=train,
        id_dict=id_dict,
        max_workers=max_workers,
    )

    # Aggregate the stats
    cleaned_stats_list = []
    for stats in nested_stats:
        stats = tally_stats(stats)
        male_female_ratio = (
            stats["sex"]["number_of_men"] / stats["sex"]["number_of_women"]
        )
        ages = np.array(stats[config.COL_AGE]["ages"])
        periods = np.array(stats["period"]["periods"])
        n_admissions = stats["admission"]["number_of_admissions"]
        inhospital_death_rate = np.round(
            stats["admission"]["number_of_discharges_expired"] / n_admissions,
            4,
        )
        los = np.array(stats["admission"]["lengths_of_stay"])
        cleaned_stats = {
            "n_patients": [stats["n_patients"]],
            "sex": {"male_to_female": [male_female_ratio]},
            "age": {
                "mean": [ages.mean()],
                "median": [np.median(ages)],
                "std": [ages.std()],
                "min": [ages.min()],
                "max": [ages.max()],
            },
            "period": {
                "mean": [periods.mean()],
                "median": [np.median(periods)],
                "std": [periods.std()],
                "min": [periods.min()],
                "max": [periods.max()],
            },
            "admission": {
                "n_admissions": [n_admissions],
                "inhospital_death_rate": [inhospital_death_rate],
                "length_of_stay": {
                    "mean": [los.mean()],
                    "median": [np.median(los)],
                    "std": [los.std()],
                    "min": [los.min()],
                    "max": [los.max()],
                },
            },
            "major_diagnoses": {k: [v] for k, v in stats["major_diagnoses"].items()},
        }
        cleaned_stats_list.append(cleaned_stats)
        bootstrapped_stats = tally_stats(cleaned_stats_list)

    return bootstrapped_stats


def ci_general_stats(bootstrapped_stats: dict, ci: float = 0.95) -> dict:
    """Compute confidence intervals from the bootstrapped results"""
    ci_dict = {}
    lower_percentile = ((1 - ci) / 2) * 100
    upper_percentile = 100 - lower_percentile

    def _compute_ci(d, ci, new_d):
        for k, v in d.items():
            if isinstance(v, list):
                array = np.array(v)
                mean = array.mean()
                lower_lim = np.percentile(array, q=lower_percentile)
                upper_lim = np.percentile(array, q=upper_percentile)
                new_v = [mean, lower_lim, upper_lim]
                new_d[k] = new_v
            elif isinstance(v, dict):
                if k not in new_d:
                    new_d[k] = {}
                new_d[k] = _compute_ci(v, ci, new_d[k])
            else:
                new_d[k] = v
        return new_d

    ci_dict = _compute_ci(bootstrapped_stats, ci, ci_dict)
    return ci_dict


def bootstrap_count_codes_in_eval_tables(
    dataset_dir: str,
    iterations: int,
    resample_size: int,
    train: bool = True,
    id_dict: dict = None,
    max_workers: int = None,
) -> pd.DataFrame:
    """Compute code prevalence in the labelled data."""
    # Bootstrap
    nested_stats = bootstrap_eval_tables(
        dataset_dir=dataset_dir,
        iterations=iterations,
        resample_size=resample_size,
        func=_count_codes_in_eval_tables,
        train=train,
        id_dict=id_dict,
        max_workers=max_workers,
    )

    # Aggregate the stats
    bootstrapped_stats = [tally_stats(s) for s in nested_stats]
    agg_dict = {"per_db": {"total": []}, "per_patient": {"total": resample_size}}
    for key in [config.DX_CODE, config.MED_CODE, config.LAB_CODE]:
        agg_dict["per_db"][key] = {"n_codes": [], "value_counts": []}
        agg_dict["per_patient"][key] = {"value_counts": []}
    for stats in bootstrapped_stats:
        agg_dict["per_db"]["total"].append(stats["per_db"]["total"])
        for key in [config.DX_CODE, config.MED_CODE, config.LAB_CODE]:
            agg_dict["per_db"][key]["n_codes"].append(stats["per_db"][key]["n_codes"])
            agg_dict["per_db"][key]["value_counts"].append(
                stats["per_db"][key]["value_counts"]
            )
            agg_dict["per_patient"][key]["value_counts"].append(
                stats["per_patient"][key]["value_counts"]
            )

    return agg_dict
