"""Module to prepare for tokenization and define the model's vocabulary from the dataset."""

from .token_handling_functions import (
    collect_unique_codes,
    create_tokenization_map,
    create_catalog,
)
from ....general_params import get_settings
from ....utils import function_wrapper


def prepare_for_tokenization():
    """Takes these steps:
    - 1. Collect unique standardized code values from training data
    - 2. Create mapping tables for tokenization
    - 3. Create catalogs for tokenization, labeling and inference

    """
    # Define parameters
    functions = [
        collect_unique_codes,
        create_tokenization_map,
        create_catalog,
    ]
    descriptions = [
        "collect unique code values",
        "create tokenization maps",
        "create catalogs",
    ]
    section_description = "Preparing for tokenization"
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
