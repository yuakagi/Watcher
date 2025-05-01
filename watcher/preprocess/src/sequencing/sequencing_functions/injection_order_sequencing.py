"""Module to align injection order records"""

import os
import pandas as pd
from .sequencing_core import sequence_core
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    render_token_map,
    convert_codes_to_indexes,
    parallel_map_partitions,
    load_categorical_dim,
    tally_stats,
)


def _sequence_injection_order_records(
    df: pd.DataFrame,
    task_no: int,
    categorical_dim: int,
    med_token_map: pd.DataFrame,
) -> pd.DataFrame:
    """Helper function that performs preprocessing steps for injection order records.

    Args:
        df (pd.DataFrame): Dataframe to be processed
        task_no (int): number of current task processed by the child process.
            This is passed by 'parallel_map_partition' function.
        categorical_dim (int): Maximum number of embedding indexes to represent a code value
        med_token_map (pd.DataFrame): Table to map medication codes into series of embedding indexes
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    stats = {}

    # ********** Categorical values to vocabulary indexes ***************
    # You need to create the 'categorical_cols', and COL_ORIGINAL_VALUE here.
    # NOTE: Replace oov values with the oov index, not tull
    df = convert_codes_to_indexes(df, med_token_map, config.COL_ITEM_CODE)
    df = df.rename(columns={config.COL_ITEM_CODE: config.COL_ORIGINAL_VALUE})
    # *******************************************************************

    # Main
    df, substats = sequence_core(
        df,
        task_no=task_no,
        categorical_dim=categorical_dim,
        type_no=config.RECORD_TYPE_NUMBERS[config.INJ_O],
        dropna_subset=[
            config.COL_PID,
            config.COL_TIMEDELTA,
            config.COL_ORIGINAL_VALUE,
            "c0",
        ],
    )

    stats = {**stats, **substats}

    return df, stats


def sequence_injection_order_records():
    """Aligns records before creating patient timeline matrices to injection order records.
    See the helper function for details.
    """
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.INJECTION_ORDER_TABLE_PATTERN
    )
    output_file_path = os.path.join(
        get_settings("SEQUENCED_TABLES_DIR"),
        config.INJECTION_ORDER_TABLE_PATTERN.replace("*", "sequenced"),
    )

    # Get the categorical_dim.
    categorical_dim = load_categorical_dim()

    # Load tokenization map.
    med_token_map = render_token_map(map_type=config.MED_CODE)

    stats_list = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_sequence_injection_order_records,
        output_file_path=output_file_path,
        chunksize=-1,
        single_file=False,
        task_no_as_arg=True,
        med_token_map=med_token_map,
        categorical_dim=categorical_dim,
    )
    return tally_stats(stats_list)


def sequence_injection_order_records_single(
    df: pd.DataFrame,
    categorical_dim: int,
    med_token_map: pd.DataFrame,
) -> pd.DataFrame:
    """Aligns records linearly before creating patient timeline matrices.
    This is designed to handle data for a single patient.
    See the helper function for details.
    """
    if df.size:
        df, _ = _sequence_injection_order_records(
            df=df,
            task_no=0,
            med_token_map=med_token_map,
            categorical_dim=categorical_dim,
        )
        return df

    else:
        return None
