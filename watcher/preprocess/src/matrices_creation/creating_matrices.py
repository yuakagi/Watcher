"""Module to create tensors from the tabular data."""

from pandas import DataFrame
from torch import Tensor
from .matrices_creation_functions import (
    create_timeline_matrices,
    create_timeline_matrix_single,
)
from ....general_params import get_settings
from ....utils import function_wrapper


def create_matrices_for_pretraining():
    """Creates tensors from the labelled aggregated record tables.

    This function creates a set of tensors for pretraining.
    """
    # Define parameters
    functions = [create_timeline_matrices]
    descriptions = ["create timeline matrices"]
    section_description = "Timeline matrices creation for pretraining"
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


def create_matrix_for_inference(
    df: DataFrame,
    categorical_dim: int,
) -> Tensor:
    """Creates a matrix for a single patient, for inference."""
    timeline_and_labels = create_timeline_matrix_single(
        df=df, categorical_dim=categorical_dim
    )
    return timeline_and_labels
