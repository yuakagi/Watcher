"""Module to align records linearly"""

from __future__ import annotations

from pandas import DataFrame
from .sequencing_functions import (
    sequence_demographics,
    sequence_admission_records,
    sequence_discharge_records,
    sequence_diagnosis_records,
    sequence_prescription_order_records,
    sequence_injection_order_records,
    sequence_lab_result_records,
    sequence_demographics_single,
    sequence_admission_records_single,
    sequence_discharge_records_single,
    sequence_diagnosis_records_single,
    sequence_prescription_order_records_single,
    sequence_injection_order_records_single,
    sequence_lab_result_records_single,
)
from ....general_params import get_settings
from ....general_params import watcher_config as config
from ....utils import function_wrapper


def sequence_data() -> None:
    """Processes records so that they align linearly.
    Each substep performs the following operations:
    - 1. Add a column 'type':
            This column is created to make it easier to identify demographic records
        when all records are combined later processes.
    - 2. Tokenize categorical values:
            Values in demographic records are tokenized.
            Because they can be represented with multiple embedding indexes, multiple columns are created
        for embedding indexes. Empty columns are padded with zeros.
    - 3. Padding
            Unused columns are padded properly.
    - 4. Drop missing values
            Ensure no missing values are left in the table.
    - 5. Preprocess timedelta objects
            Extract year, month, day, hour and minute components from all timedelta objects and
        create columns for each of them.
            To ensure all demographic records come at first in patient timelines,
        all of these values are filled with zeros for demographic records.
    - 6. Organize records for patient timeline matrices
        Re-arrange records to create the basis for the matrices.
        The process is complex, see 'align_records_linearly' for details.

    """
    # Define parameters
    functions = [
        sequence_demographics,
        sequence_admission_records,
        sequence_discharge_records,
        sequence_diagnosis_records,
        sequence_prescription_order_records,
        sequence_injection_order_records,
        sequence_lab_result_records,
    ]
    descriptions = [
        "create a sequence of demographic data",
        "create a sequence of admission records",
        "create a sequence of discharge records",
        "create a sequence of diagnosis records",
        "create a sequence of prescription order records",
        "create a sequence of injection order records",
        "create a sequence of laboratory test results",
    ]
    section_description = "Data preprocess"
    sheet_pattern = get_settings("PERFORMANCE_SHEET_PTN")
    json_path = sheet_pattern.replace(
        "*", "_".join(section_description.lower().split(" "))
    )
    for function, description in zip(functions, descriptions):
        function_wrapper(
            function=function,
            section_description=section_description,
            step_description=description,
            json_path=json_path,
        )

    print("All data preprocessing is completed.")


def sequence_data_single(
    tables: tuple[DataFrame, ...], model: Watcher
) -> tuple[DataFrame]:
    """Aligns records from a patient for model inference.

    Returns:
        tuple[DataFrame]: A tuple of processed DataFrames for different records.
    """

    # Pick up objects
    special_token_dict = model.interpreter.special_token_dict
    n_numeric_bins = model.n_numeric_bins
    percentiles = model.interpreter.percentiles_for_preproc  # <= 'for_preproc'
    percentile_cols = model.interpreter.percentile_cols
    num_stats = model.interpreter.num_stats_for_preproc  # <= 'for_preproc'
    nonnum_stats = model.interpreter.nonnum_stats
    categorical_dim = model.categorical_dim
    categorical_cols = [f"c{i}" for i in range(categorical_dim)]
    token_map_cols = [config.COL_ORIGINAL_VALUE] + categorical_cols
    med_token_map = model.interpreter.catalogs[config.MED_CODE][token_map_cols].copy()
    diagnosis_token_map = model.interpreter.catalogs[config.DX_CODE][
        token_map_cols
    ].copy()
    lab_token_map = model.interpreter.catalogs[config.LAB_CODE][token_map_cols].copy()
    adm_index = special_token_dict["[ADM]"]
    dsc_index = special_token_dict["[DSC]"]

    # Process each table
    dmg_df = sequence_demographics_single(
        tables[0],
        categorical_dim=categorical_dim,
        special_token_dict=special_token_dict,
    )
    adm_df = sequence_admission_records_single(
        df=tables[1],
        adm_index=adm_index,
        categorical_dim=categorical_dim,
    )
    dsc_df = sequence_discharge_records_single(
        df=tables[2],
        dsc_index=dsc_index,
        categorical_dim=categorical_dim,
        special_token_dict=special_token_dict,
    )
    dx_df = sequence_diagnosis_records_single(
        df=tables[3],
        categorical_dim=categorical_dim,
        diagnosis_token_map=diagnosis_token_map,
    )
    presc_order_df = sequence_prescription_order_records_single(
        df=tables[4],
        categorical_dim=categorical_dim,
        med_token_map=med_token_map,
    )
    injec_order_df = sequence_injection_order_records_single(
        df=tables[5],
        categorical_dim=categorical_dim,
        med_token_map=med_token_map,
    )
    lab_result_df = sequence_lab_result_records_single(
        df=tables[6],
        categorical_dim=categorical_dim,
        n_numeric_bins=n_numeric_bins,
        percentiles=percentiles,
        percentile_cols=percentile_cols,
        nonnum_stats=nonnum_stats,
        num_stats=num_stats,
        lab_token_map=lab_token_map,
        special_token_dict=special_token_dict,
    )

    # Update `tables`
    # NOTE: Empty dataframes are replaced with None
    tables = (
        dmg_df,
        adm_df,
        dsc_df,
        dx_df,
        presc_order_df,
        injec_order_df,
        lab_result_df,
    )

    return tables
