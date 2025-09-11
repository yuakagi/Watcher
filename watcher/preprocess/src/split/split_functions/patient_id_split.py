"""Functions to split patient IDs"""

import os
import json
import random
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import parallel_map_partitions, tally_stats


def _split_patient_id_temporal(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
) -> dict:
    """Helper function that creates patient ID lists for train, val and test datasets.

    Args:
        df (pd.DataFrame): Dataframe that contain all unique patient IDs.
        train_size (float): Any arbitrary number from 0 .0 to 1.0
        val_size (float): Any arbitrary number from 0 .0 to 1.0
            Note: There is no 'test_size' argument.
                The test size is calculated by 1 - (train_size + val_size).
                Ensure that the sum of train_size and val_size is smaller than 1.0.
    Returns:
        split_id_dict (dict): Dictionary that store the three patient ID lists.
            Keys are config.TRAIN ('train'), config.VAL ('validation') and config.TEST ('test').
            Each key are paired with a corresponding patient ID list.
    """
    # Set seed
    random.seed(config.SEED)

    # Initialize variables
    train_val_ids = []
    test_ids = []
    train_period_start = get_settings("TRAIN_PERIOD_START")
    train_period_end = get_settings("TRAIN_PERIOD_END")
    test_period_start = get_settings("TEST_PERIOD_START")
    test_period_end = get_settings("TEST_PERIOD_END")

    # *************
    # * Exclusion *
    # *************
    # Exclude patients without visiting dates
    exclusion_mask = (
        df[config.COL_LAST_VISIT_DATE].isna() | df[config.COL_FIRST_VISIT_DATE].isna()
    )
    # Exclude patients whose last visiting dates come before the start of the training period
    exclusion_mask = df[config.COL_LAST_VISIT_DATE] < train_period_start
    # Exclude patients whose first visiting dates come after the end of the testing period
    exclusion_mask = exclusion_mask | (
        df[config.COL_FIRST_VISIT_DATE] > test_period_end
    )
    # Exclude patients whose records exist only between the two periods
    exclusion_mask = exclusion_mask | (
        (df[config.COL_FIRST_VISIT_DATE] > train_period_end)
        & (df[config.COL_LAST_VISIT_DATE] < test_period_start)
    )
    # Exclude patinets invalid both for training and testing
    excluded_df = df.loc[~exclusion_mask, :]

    # ***************************
    # * Train vs test-val split *
    # ***************************
    # Define sizes
    total_patients = len(excluded_df)
    test_size = 1 - train_size - val_size
    n_test = int(total_patients * test_size)

    # Patients whose last visiting dates come before the start of the testing period are all used for training or validation
    train_val_mask = excluded_df[config.COL_LAST_VISIT_DATE] < test_period_start
    train_val_ids += excluded_df.loc[train_val_mask, config.COL_PID].tolist()

    # Patients whose first visiting dates come after the start of the testing period are all used for testing
    test_mask = excluded_df[config.COL_FIRST_VISIT_DATE] >= test_period_start
    test_ids += excluded_df.loc[test_mask, config.COL_PID].tolist()

    # Select IDs from the rest of IDs
    residual_mask = ~(train_val_mask | test_mask)
    residual_df = excluded_df[residual_mask].copy()
    # Sort patients so that newer patients come first
    residual_df = residual_df.sort_values(
        config.COL_FIRST_VISIT_DATE, ascending=False
    ).reset_index(drop=True)
    residual_ids = residual_df[config.COL_PID].tolist()
    # Finish splitting if the test size is already met
    if len(test_ids) >= n_test:
        # Put all the other IDs to the training list
        residual_ids = excluded_df.loc[residual_mask, config.COL_PID].to_list()
        train_val_ids += residual_ids
    # If test-val size has not been met yet, pick up other candidates
    else:
        # Caluculate number of patients needed to achive the target test size
        test_cap = n_test - len(test_ids)
        # If adding all the residual patient IDs is not enought to achive the target test size, add them all to the test ID list
        if len(residual_df) <= test_cap:
            test_ids += residual_ids
        else:
            # Add newer patients to the test-val set
            test_ids += residual_ids[:test_cap]
            # Put the rest of the candidates to the training set
            train_val_ids += residual_ids[test_cap:]

    # **********************
    # * Train vs val split *
    # **********************
    # Calculate the number of patients for training and validation
    proportion_train = train_size / (train_size + val_size)
    n_train = int(len(train_val_ids) * proportion_train)
    n_val = len(train_val_ids) - n_train
    # Randomly select ids for validation
    val_ids = random.sample(train_val_ids, n_val)
    train_ids = list(set(train_val_ids) - set(val_ids))
    # Assign
    split_id_dict = {
        config.TRAIN: train_ids,
        config.VAL: val_ids,
        config.TEST: test_ids,
    }

    return split_id_dict


def split_patient_id_temporal() -> None:
    """Splits all patient IDs into train, val, and test sets.

    NOTE: This process works if patient demographic records from a data source are saved separately
    in multiple files; however, demographic records should be saved in a single file to ensure
    the precision of the splitting ratio.

    Returns:
        None
    """

    # Patient IDs are extracted from demographic tables.
    source_path = os.path.join(
        get_settings("VISITING_DATES_DIR"), config.VISITING_DATE_TABLE
    )
    # Get the environment variables
    output_dir = get_settings("PID_DIR")

    # Perform splitting
    id_dict = _split_patient_id_temporal(
        df=pd.read_pickle(source_path),
        train_size=get_settings("TRAIN_SIZE"),
        val_size=get_settings("VAL_SIZE"),
    )

    # Shuffle IDs
    random.seed(config.SEED)
    for flag in id_dict.keys():
        random.shuffle(id_dict[flag])

    # Save
    with open(
        os.path.join(output_dir, config.TOTAL_PATIENT_ID_LIST), "w", encoding="utf-8"
    ) as f:
        json.dump(id_dict, f, indent=2)
