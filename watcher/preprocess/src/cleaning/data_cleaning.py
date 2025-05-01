"""The main module for data cleaning."""

from datetime import datetime
from pandas import DataFrame
from .cleaning_functions import clean_records_for_dataset, clean_records_for_inference
from ....utils import function_wrapper
from ....general_params import get_settings


def clean_data():
    """Cleans record tables individually by record types."""
    # Define parameters
    functions = [clean_records_for_dataset]
    descriptions = ["clean all clinical records"]
    section_description = "Data cleaning"
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


def clean_data_single(
    patient_id: str,
    start: str,
    end: str,
    db_schema: str = "public",
) -> tuple[tuple[DataFrame, ...], datetime]:
    """Cleans records from a patient for model inference."""
    tables, dob = clean_records_for_inference(
        patient_id=patient_id,
        start=start,
        end=end,
        db_schema=db_schema,
    )
    return tables, dob
