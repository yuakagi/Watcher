"""Module to align discharge records"""

import os
import pandas as pd
from .sequencing_core import sequence_core
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    parallel_map_partitions,
    load_special_token_index,
    load_categorical_dim,
    load_special_token_dict,
    tally_stats,
)


def _sequence_discharge_records(
    df: pd.DataFrame,
    task_no: int,
    dsc_index: int,
    categorical_dim: int,
    special_token_dict: pd.DataFrame,
) -> pd.DataFrame:
    """Performs preprocessing steps for discharge records.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
        task_no (int): number of current task processed by the child process.
            This is passed by 'parallel_map_partition' function.
        dsc_index (int): Embedding index for [DSC] token.
        categorical_dim (int): Maximum number of embedding indexes to represent a code value.
        special_token_dict (dict): Dictionary to convert discharge disposition tokens into embedding indexes.
    Returns:
        df (pd.DataFrame): Processed dataframe.
    """
    stats = {}

    # ********** Categorical values to vocabulary indexes ***************
    # You need to create the 'categorical_cols', and COL_ORIGINAL_VALUE here.
    # NOTE: Replace oov values with the oov index, not tull
    df["c0"] = dsc_index
    df[config.COL_ORIGINAL_VALUE] = "[DSC]"
    # *******************************************************************

    # ********** Nonnumeric values to vocabulary indexes ****************
    # You need to create the COL_NONNUMERIC, and CCOL_ORIGINAL_NONNUMERIC here.
    # NOTE: Replace oov values with the oov index, not null.
    df["disposition_token"] = df["disposition"].replace(
        config.DISCHARGE_STATUS_TOKENS, regex=True
    )
    special_token_df = pd.DataFrame.from_dict(
        special_token_dict, orient="index", columns=[config.COL_NONNUMERIC]
    )
    df = df.merge(
        special_token_df,
        left_on="disposition_token",
        right_index=True,
        how="left",
    )
    df[config.COL_NONNUMERIC] = df[config.COL_NONNUMERIC].fillna(0)
    df[config.COL_ORIGINAL_NONNUMERIC] = df["disposition_token"].copy()
    # *******************************************************************

    # Main
    df, substats = sequence_core(
        df,
        task_no=task_no,
        categorical_dim=categorical_dim,
        type_no=config.RECORD_TYPE_NUMBERS[config.DSC],
        dropna_subset=[config.COL_PID, config.COL_TIMEDELTA],
    )

    stats = {**stats, **substats}

    return df, stats


def sequence_discharge_records() -> dict:
    """Aligns records linearly before creating patient timeline matrices to discharge records.
    See the helper function for details.
    """
    source_path_pattern = os.path.join(
        get_settings("CLEANED_TABLES_DIR"), config.DISCHARGE_TABLE_PATTERN
    )
    output_file_path = os.path.join(
        get_settings("SEQUENCED_TABLES_DIR"),
        config.DISCHARGE_TABLE_PATTERN.replace("*", "sequenced"),
    )
    # Get the index of [DSC] token.
    dsc_index = load_special_token_index("[DSC]")
    categorical_dim = load_categorical_dim()
    special_token_dict = load_special_token_dict()

    stats_list = parallel_map_partitions(
        source_path_pattern=source_path_pattern,
        function=_sequence_discharge_records,
        output_file_path=output_file_path,
        chunksize=-1,
        single_file=False,
        task_no_as_arg=True,
        dsc_index=dsc_index,
        categorical_dim=categorical_dim,
        special_token_dict=special_token_dict,
    )
    return tally_stats(stats_list)


def sequence_discharge_records_single(
    df: pd.DataFrame,
    dsc_index: int,
    categorical_dim: int,
    special_token_dict: pd.DataFrame,
) -> pd.DataFrame:
    """Aligns records linearly before creating patient timeline matrices.
    This is designed to handle data for a single patient.
    See the helper function for details.
    """
    if df.size:
        df, _ = _sequence_discharge_records(
            df=df,
            task_no=0,
            dsc_index=dsc_index,
            categorical_dim=categorical_dim,
            special_token_dict=special_token_dict,
        )
        return df

    else:
        return None
