"""Module to align demographic records"""

import os
from datetime import timedelta
import pandas as pd
from .sequencing_core import sequence_core
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    parallel_map_partitions,
    load_categorical_dim,
    load_special_token_dict,
    tally_stats,
)


def _sequence_demographics(
    df: pd.DataFrame,
    task_no: int,
    categorical_dim: int,
    special_token_dict: dict,
) -> pd.DataFrame:
    """Helper function that performs preprocessing steps for demographic records.
    Args:
        df (pd.DataFrame): Dataframe to be processed
        task_no (int): number of current task processed by the child process.
            This is passed by 'parallel_map_partition' function.
        categorical_dim (int): Maximum number of embedding indexes to represent a code value
        special_token_dict (dict): Dictionary to convert demographic value tokens into embedding indexes
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    stats = {}
    # Add timestamp column for consistency
    df[config.COL_TIMESTAMP] = ""
    # Ensure that demographic cata comes first when sorted by timedelta.
    df[config.COL_TIME_AVAILABLE] = timedelta(seconds=0)
    df[config.COL_TIMEDELTA] = timedelta(seconds=0)

    # ********** Nonnumeric values to vocabulary indexes ****************
    # You need to create the COL_NONNUMERIC, and CCOL_ORIGINAL_NONNUMERIC here.
    # NOTE: Replace oov values with the oov index, not null.
    df[config.COL_SEX] = df[config.COL_SEX].replace(config.SEX_TOKENS)
    df[config.COL_ORIGINAL_NONNUMERIC] = df[config.COL_SEX].copy()
    special_token_df = pd.DataFrame.from_dict(
        special_token_dict, orient="index", columns=[config.COL_NONNUMERIC]
    )
    df = df.merge(
        special_token_df, left_on=config.COL_SEX, right_index=True, how="left"
    )
    df[config.COL_NONNUMERIC] = df[config.COL_NONNUMERIC].fillna(config.OOV_INDEX)
    # *******************************************************************

    # Main
    df, substats = sequence_core(
        df=df,
        task_no=task_no,
        categorical_dim=categorical_dim,
        type_no=config.RECORD_TYPE_NUMBERS[config.DMG],
        dropna_subset=[config.COL_PID, config.COL_NONNUMERIC],
    )
    stats = {**stats, **substats}

    return df, stats


def sequence_demographics() -> dict:
    """Aligns records linearly before creating patient timeline matrices to demographic records.
    See the helper function for details.
    """
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.DEMOGRAPHIC_TABLE_PATTERN
    )
    output_file_path = os.path.join(
        get_settings("SEQUENCED_TABLES_DIR"),
        config.DEMOGRAPHIC_TABLE_PATTERN.replace("*", "sequenced"),
    )

    # Get the categorical_dim and the special token reference.
    categorical_dim = load_categorical_dim()
    special_token_dict = load_special_token_dict()

    stats_list = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_sequence_demographics,
        output_file_path=output_file_path,
        chunksize=-1,
        single_file=False,
        task_no_as_arg=True,
        categorical_dim=categorical_dim,
        special_token_dict=special_token_dict,
    )
    return tally_stats(stats_list)


def sequence_demographics_single(
    df: pd.DataFrame,
    categorical_dim: int,
    special_token_dict: dict,
) -> pd.DataFrame:
    """Aligns records linearly before creating patient timeline matrices.
    This is designed to handle data for a single patient.
    See the helper function for details.
    """
    if df.size:
        df, _ = _sequence_demographics(
            df=df,
            task_no=0,
            categorical_dim=categorical_dim,
            special_token_dict=special_token_dict,
        )
        return df

    else:
        return None
