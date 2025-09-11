"""Main module for patient ID split"""

from .split_functions import collect_visiting_dates, split_patient_id_temporal
from ....general_params import get_settings
from ....utils import function_wrapper


def find_visiting_dates():
    """Collects visiting dates of patients"""
    section_description = "Collecting visiting dates"
    sheet_pattern = get_settings("PERFORMANCE_SHEET_PTN")
    json_path = sheet_pattern.replace(
        "*", "_".join(section_description.lower().split(" "))
    )
    function_wrapper(
        function=collect_visiting_dates,
        section_description=section_description,
        step_description="collect visiting dates",
        json_path=json_path,
    )


def split_patient_ids():
    """Splits patient IDs, and save them as json files.
    See 'split_patient_id_temporal' for details
    """
    section_description = "Patient ID splitting"
    sheet_pattern = get_settings("PERFORMANCE_SHEET_PTN")
    json_path = sheet_pattern.replace(
        "*", "_".join(section_description.lower().split(" "))
    )
    function_wrapper(
        function=split_patient_id_temporal,
        section_description=section_description,
        step_description="split patient IDs",
        json_path=json_path,
    )
