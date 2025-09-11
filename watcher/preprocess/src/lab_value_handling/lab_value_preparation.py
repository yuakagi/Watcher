"""Main module for laboratory test value handlings"""

from .lab_value_handling_functions import tally_lab_values
from ....general_params import get_settings
from ....utils import function_wrapper


def prepare_for_lab_values():
    """Prepares for laboratory value preprocessing."""
    # Define parameters
    functions = [tally_lab_values]
    descriptions = ["tally numeric and nonnumeric values"]
    section_description = "Cleaning and tallying lab values"
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
