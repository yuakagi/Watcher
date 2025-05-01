"""Module to put labels to the aggregated records for training."""

from .eval_table_creation_functions import prepare_eval_tables
from ....general_params import get_settings
from ....utils import function_wrapper


def make_eval_tables():
    """Creates evaluation tables for performance assessment"""
    # Define parameters
    functions = [prepare_eval_tables]
    descriptions = ["Create tables for task performance evaluation"]
    section_description = "Eval table making"
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
