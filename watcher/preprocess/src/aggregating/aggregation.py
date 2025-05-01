"""Module to aggregate records from different tables into a single table."""

from pandas import DataFrame
from .aggregation_functions import (
    aggregate_records,
    aggregate_records_single,
    finalize_aggregated_files,
    finalize_aggregated_single,
)
from ....general_params import get_settings
from ....utils import function_wrapper


def aggregate_data():
    """Aggregates records of different data types by patients

    See helper functions for process details.
    """
    # Define parameters
    functions = [
        aggregate_records,
        finalize_aggregated_files,
    ]
    descriptions = [
        "aggregate records",
        "finalize aggregated files",
    ]
    section_description = "Data aggregation"
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


def aggregate_data_single(categorical_dim: int, tables: tuple[DataFrame]) -> DataFrame:
    """Aggregate all tables, and finalize the aggregated table for model inference."""
    agg_df = aggregate_records_single(tables=tables)
    agg_df = finalize_aggregated_single(agg_df=agg_df, categorical_dim=categorical_dim)
    return agg_df
