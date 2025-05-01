"""Module to align admission records"""

import os
import pandas as pd
from .sequencing_core import sequence_core
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    load_categorical_dim,
    load_special_token_index,
    parallel_map_partitions,
    tally_stats,
)


def _sequence_admission_records(
    df: pd.DataFrame,
    task_no: int,
    adm_index: int,
    categorical_dim: int,
) -> pd.DataFrame:
    """Performs preprocessing steps for admission records.

    Args:
        df (pd.DataFrame): Dataframe to be processed
        task_no (int): number of current task processed by the child process.
            This is passed by 'parallel_map_partition' function.
        adm_index (int): Embedding index for [ADM] token
        categorical_dim (int): Maximum number of embedding indexes to represent a code value
    Returns:
        df (pd.DataFrame): Processed dataframe
    """
    stats = {}
    # ********** Categorical values to vocabulary indexes ***************
    # You need to create the 'categorical_cols', and COL_ORIGINAL_VALUE here.
    # NOTE: Replace oov values with the oov index, not tull
    df["c0"] = adm_index
    df[config.COL_ORIGINAL_VALUE] = "[ADM]"
    # *******************************************************************

    # Main
    df, substats = sequence_core(
        df,
        task_no=task_no,
        categorical_dim=categorical_dim,
        type_no=config.RECORD_TYPE_NUMBERS[config.ADM],
        dropna_subset=[config.COL_PID, config.COL_TIMEDELTA],
    )

    stats = {**stats, **substats}

    return df, stats


def sequence_admission_records() -> dict:
    """Aligns records linearly before creating patient timeline matrices to admission records.
    See the helper function for details.
    """
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.ADMISSION_TABLE_PATTERN
    )
    output_file_path = os.path.join(
        get_settings("SEQUENCED_TABLES_DIR"),
        config.ADMISSION_TABLE_PATTERN.replace("*", "sequenced"),
    )

    # Get the index of [ADM] token and categorical_dim.
    adm_index = load_special_token_index("[ADM]")
    categorical_dim = load_categorical_dim()

    stats_list = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        output_file_path=output_file_path,
        function=_sequence_admission_records,
        chunksize=-1,
        single_file=False,
        task_no_as_arg=True,
        adm_index=adm_index,
        categorical_dim=categorical_dim,
    )

    return tally_stats(stats_list)


def sequence_admission_records_single(
    df: pd.DataFrame,
    adm_index: int,
    categorical_dim: int,
) -> pd.DataFrame:
    """Aligns records linearly before creating patient timeline matrices.
    This is designed to handle data for a single patient.
    See the helper function for details.
    """
    if df.size:
        df, _ = _sequence_admission_records(
            df=df,
            task_no=0,
            adm_index=adm_index,
            categorical_dim=categorical_dim,
        )
        return df

    else:
        return None
