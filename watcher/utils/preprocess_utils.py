"""Utils for preprocessing"""

from datetime import timedelta, datetime
from typing import Literal
import pandas as pd
import numpy as np
import torch
from .general_utils import (
    get_categorical_cols,
    load_catalog,
    load_special_token_index,
    load_db_engine,
)
from ..general_params import watcher_config as config
from ..general_params import get_settings


def format_df(
    df: pd.DataFrame,
    table_params: dict,
    drop_cols: bool = True,
    errors: Literal["coerce", "raise"] = "coerce",
) -> pd.DataFrame:
    """
    Formats a table using a table definition dictionary. Unnecessary columns are dropped.
    Args:
        df (DataFrame): Table to be examined.
        table_params (dictionary): Dictionary with its keys as column names, and its values as data types.
        drop_cols (bool): If true, this function drops all other columns that are not in the dtype dict.
        errors (Literal["coerce", "raise"]): Error handlings during dtype conversion.
    Returns:
        df (DataFrame): Cleaned table.
    """
    final_cols = []
    for c, t in table_params.items():
        if c in df.columns:
            final_cols.append(c)
            if t == timedelta:
                if not pd.api.types.is_timedelta64_dtype(df[c]):
                    df[c] = pd.to_timedelta(df[c], errors=errors)
            elif t == datetime:
                if not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors=errors)
            elif t == float:
                if not pd.api.types.is_float_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors=errors).astype(float)
            elif t == int:
                if not pd.api.types.is_integer_dtype(df[c]):
                    # With null values in column
                    # NOTE: This converts to 'Int64' (catpital 'I') to enable nullable intger type.
                    #       This is a compromise. In general, you should avoid mixing null values in interger columns.
                    if df[c].isna().any():
                        df[c] = df[c].astype("Int64")
                    # Without null values
                    # NOTE: This converts the column to 'int64', the default integer type.
                    else:
                        df[c] = df[c].astype(float).astype(int)
            elif t == str:
                if not pd.api.types.is_string_dtype(df[c]):
                    # NOTE: Null values (such as NaN) are converted to strings like 'nan' by astype(str);
                    #       therefore, masking operation is needed
                    missings = df[c].isna()
                    df[c] = df[c].astype(str).mask(missings, None)

            # All other dtypes
            # NOTE: no dtypes other than listed above are expected to be used during preprocessing.
            #       Add other operations if necessary.
            else:
                df[c] = df[c].astype(t)
    if drop_cols:
        df = df[final_cols]
    return df


def render_token_map(map_type: str) -> pd.DataFrame:
    """Returns a table to map categorical or code values to series of embedding indexes.
    Upon returning the mapping table, values in categorical columns ('c*' columns) are made into integers.
    Args:
        map_type (str): Type of mapping table. See 'load_catalog' for available choices.
    Returns:
        token_map (pd.DataFrame): Table for mapping.
            It has the column 'config.COL_ORIGINAL_VALUE' for mapping key, and
            the multiple of columns with the 'c*' column name pattern, each of which stores embedding indexes.
    """
    # Load catalog
    token_map = load_catalog(
        catalog_type=map_type,
        catalogs_dir=get_settings("CATALOGS_DIR"),
    )
    # Drop unnecessary columns
    categorical_cols = get_categorical_cols(token_map)
    final_cols = [config.COL_ORIGINAL_VALUE] + sorted(categorical_cols)
    token_map = token_map[final_cols]

    return token_map


def add_distribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds columns to separate records by temporal distribions.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
    Returns:
        df (pd.DataFrame): Processed dataframe.
    """
    # Get environment variables
    train_start = get_settings("TRAIN_PERIOD_START")
    train_end = get_settings("TRAIN_PERIOD_END")
    test_start = get_settings("TEST_PERIOD_START")
    test_end = get_settings("TEST_PERIOD_END")

    # Add columns to identify temporal distribution
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])
    train_period_mask = df[config.COL_TIMESTAMP].between(train_start, train_end)
    test_period_mask = df[config.COL_TIMESTAMP].between(test_start, test_end)
    df.loc[:, config.COL_TRAIN_PERIOD] = train_period_mask.astype(int)
    df.loc[:, config.COL_TEST_PERIOD] = test_period_mask.astype(int)

    return df


def add_update_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a column for update period.

    Args:
        df (pd.DataFrame): Dataframe to be processed.
    Returns:
        df (pd.DataFrame): Processed dataframe.
    """
    # Get environment variables
    update_start = get_settings("UPDATE_PERIOD_START")
    update_end = get_settings("UPDATE_PERIOD_END")

    # Add columns to identify temporal distribution
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])
    update_period_mask = df[config.COL_TIMESTAMP].between(update_start, update_end)
    df.loc[:, config.COL_UPDATE_PERIOD] = update_period_mask.astype(int)

    return df


def convert_codes_to_indexes(
    df: pd.DataFrame,
    token_map: pd.DataFrame,
    left_on: str,
    right_on=config.COL_ORIGINAL_VALUE,
) -> pd.DataFrame:
    """Adds a column of embedding index sequences from a column of coded values.
    Args:
        df (pd.DataFrame): Dataframe to be processed.
        token_map (pd.DataFrame): Table used to map embedding sequences.
        left_on (str): Name of the column that contain the target coded values in the dataframe
            to be processed.
        right_on (str): Name of the column that contain the target coded values in the mapping table.
    Returns:
        df (pd.DataFrame): Processed dataframe.
    """
    df = df.merge(
        token_map,
        left_on=left_on,
        right_on=right_on,
        how="left",
    )
    df = df.drop(right_on, axis=1)
    # Handling out-of-vocabulary entities
    oov_mask = (df[left_on].fillna("") != "") & (df["c0"].isna())
    categorical_cols = get_categorical_cols(df)
    df.loc[oov_mask, categorical_cols] = 0
    df.loc[oov_mask, "c0"] = config.OOV_INDEX

    return df


def load_lab_percentiles(
    single_unit: bool, file_path: str | None = None
) -> tuple[pd.DataFrame, list]:
    """Loads a table of pecentiles.
    Args:
        single_unit (bool): If true, only the most frequently seen units are selected so that only one unit is left for a
            unique laboratory test code.
        file_path (str): Path to the file. If not specified, the path is loaded from the environment variable.
    Returns:
        percentiles (pd.DataFrame): Loaded numeric stats table.
        percentile_cols (list): Columns names for percentiles.
    """
    # Define data types for specific columns
    dtypes = {
        config.COL_ITEM_CODE: str,
        "unit": str,
        "count": int,
    }

    # Read CSV file
    if file_path is None:
        file_path = get_settings("NUMERIC_PERCENTILES_PTN").replace("*", "train")
    percentiles = pd.read_csv(
        file_path,
        header=0,
        na_values=config.NA_VALUES,
        dtype=dtypes,
    )

    # Set all other columns to float
    # NOTE: The original column order MUST be preserved.
    percentile_cols = [col for col in percentiles.columns if col not in dtypes.keys()]
    percentiles[percentile_cols] = percentiles[percentile_cols].astype(float)

    # Select most frequent units only
    if single_unit:
        percentiles = percentiles.sort_values(
            [config.COL_ITEM_CODE, "count"], ascending=[True, False]
        ).drop_duplicates(subset=[config.COL_ITEM_CODE], keep="first")

    return percentiles, percentile_cols


def load_numeric_stats(single_unit: bool, file_path: str | None = None) -> pd.DataFrame:
    """Loads numeric stats."""
    # Define data types
    dtypes = {
        config.COL_ITEM_CODE: str,
        "unit": str,
        "count": int,
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
    }
    # Read csv
    if file_path is None:
        file_path = get_settings("NUMERIC_STATS_PTN").replace("*", "train")
    num_stats = pd.read_csv(
        file_path,
        header=0,
        na_values=config.NA_VALUES,
        usecols=list(dtypes.keys()),
        dtype=dtypes,
    )

    # Select most frequent units only
    if single_unit:
        num_stats = num_stats.sort_values(
            [config.COL_ITEM_CODE, "count"], ascending=[True, False]
        ).drop_duplicates(subset=[config.COL_ITEM_CODE], keep="first")

    return num_stats


def load_nonnumeric_stats(file_path: str = None) -> pd.DataFrame:
    """Loads a table of nonnumeric test result value stats.
    Args:
        file_path (str): Path to the file. If not specified, the path is loaded from the environment variable.
    Returns:
        nonnum_stats(pd.DataFrame): Loaded numeric stats table.
            Columns:
                config.COL_ITEM_CODE: laboratory test codes
                'token': Tokenized categorical test result values
                config.COL_NONNUMERIC: Unique nonnumeric test tesult values
                'unique count': Number of unique test result values for each laboratory test code
                'count': Number of laboratory test-value pairs seen in the train data
    """
    # Load lab nonnumeric stats.
    if file_path is None:
        file_path = get_settings("NONNUMERIC_STATS_PTN").replace("*", "train")
    nonnum_stats = pd.read_csv(file_path, header=0, na_values=config.NA_VALUES)
    # Check data types
    nonnum_stats = nonnum_stats.astype(
        {
            config.COL_ITEM_CODE: str,
            config.COL_TOKEN: str,
            config.COL_NONNUMERIC: str,
            "unique_count": int,
            "count": int,
        }
    )
    return nonnum_stats


def preprocess_timedelta_series(
    timedelta_series: pd.Series,
) -> pd.DataFrame:
    """Extracts years, months, days, hours and minutes from timedelta objects and normalizes them.
    Args:
        timedelta_series (pd.Series): Series of timedelta objects.
    Returns:
        td_componet_df (pd.Dataframe): Dataframe with the columns of years, months, days, hours and minutes.
            Each of the rows contains normalized timedelta components.
            Its indexes are shared with the input series.
    """
    # Compute total days and seconds
    days = timedelta_series.dt.days
    seconds = timedelta_series.dt.seconds

    # Calculate years, months, days, hours, and minutes then normalize (0~1) them
    # NOTE: A 'month' is defined as 30 days here. Therefore, months in patient age can be up to 12.
    #       A 'year' is defined as 365 days, which does not consider leap years. Therefore, patient ages, especially old ages, can be several days older than actual ages.
    col_years = (
        days // 365 / 120
    )  # <- Can be > 1.0, if the patient is older than 120 yo.
    col_months = days % 365 // 30 / 12
    col_days = days % 365 % 30 / 29
    col_hours = seconds // (60 * 60) / 23
    col_minutes = seconds % (60 * 60) // 60 / 59
    td_componet_df = pd.concat(
        [col_years, col_months, col_days, col_hours, col_minutes], axis=1
    )
    td_componet_df.columns = config.TIMEDELTA_COMPONENT_COLS

    return td_componet_df


def preprocess_timedelta_df(
    df: pd.DataFrame, timedelta_col: str = config.COL_TIMEDELTA
) -> pd.DataFrame:
    """Adds columns of years, months, days, hours and minutes extracted from timedelta objects.
    The extracted values are made into distinct columns.
    Args:
        df(pd.DataFrame): Dataframe to be processed
        timedelta_col(str): Column that contains timedelta objects
    Returns:
        df(pd.DataFrame): Processed dataframe
    """
    df[timedelta_col] = pd.to_timedelta(df[timedelta_col], unit=None)
    td_component_df = preprocess_timedelta_series(df[timedelta_col])
    df = pd.concat([df, td_component_df], axis=1)

    return df


def map_code_to_name(
    codes: pd.Series,
    # pylint: disable=reportInvalidTypeForm
    code_type: Literal[config.DX_CODE, config.MED_CODE, config.LAB_CODE],
) -> np.ndarray:
    """Maps codes to names.
    Args:
        codes (pd.Series): Series of codes.
            Provisional diagnoses codes must end with the suffix for provisional diagnosis (config.PROV_SUFFIX)
            with out a space. (For example, S060(prov.))
        append_code (bool, optional): If true, the original diagnosis codes are appended after tests.
    """
    # Code map path list
    map_names = {
        config.DX_CODE: config.TB_DX_CODES,
        config.MED_CODE: config.TB_MED_CODES,
        config.LAB_CODE: config.TB_LAB_CODES,
    }
    # Make the input series as a dataframe for merging operation
    codes_df = pd.DataFrame(codes)
    codes_df.columns = [config.COL_ITEM_CODE]

    # Get the map
    table = map_names[code_type]
    schema = get_settings("DB_SCHEMA")
    engine = load_db_engine()
    code_map = pd.read_sql(
        f"SELECT {config.COL_ITEM_CODE}, {config.COL_ITEM_NAME} from {schema}.{table}",
        con=engine,
    )
    code_map = code_map.dropna(how="any")
    if config.ISOLATE_PROVISIONAL_DIAGNOSES and (code_type == config.DX_CODE):
        code_map_prv = code_map.copy()
        # Append the suffix for provisional diagnosis
        code_map_prv[config.COL_ITEM_CODE] = (
            code_map_prv[config.COL_ITEM_CODE] + config.PROV_SUFFIX
        )
        code_map_prv[config.COL_ITEM_NAME] = (
            code_map_prv[config.COL_ITEM_NAME] + " " + config.PROV_SUFFIX
        )
        code_map = pd.concat([code_map, code_map_prv], ignore_index=True)

    # Mapping codes to names
    codes_df = codes_df.merge(code_map, how="left", on=config.COL_ITEM_CODE)
    # Filling missing names with codes
    codes_df[config.COL_ITEM_NAME] = codes_df[config.COL_ITEM_NAME].fillna(
        codes_df[config.COL_ITEM_CODE]
    )
    # Returns the translated names
    translated = codes_df[config.COL_ITEM_NAME].values

    return translated


def map_special_tokens_to_name(token_codes: pd.Series) -> np.ndarray:
    """Translate special tokens into English."""
    # Make the input series as a dataframe for merging operation
    token_codes = pd.DataFrame(token_codes)
    token_codes.columns = [config.COL_CODE]

    # Load medication-to-text mapping table
    token_to_text = pd.DataFrame(
        {
            config.COL_CODE: config.TOKEN_DESCRIPTIONS.keys(),
            config.COL_TEXT: config.TOKEN_DESCRIPTIONS.values(),
        }
    )

    # Map tokens to texts
    token_codes = token_codes.merge(token_to_text, how="left", on=config.COL_CODE)
    token_codes = token_codes.fillna("")

    translated = token_codes[config.COL_TEXT].values
    return translated


def create_labels_from_timedelta_pairs(
    timedelta_series: pd.Series,
    series_to_copmare: pd.Series,
    first_timedelta_label: int,
    small_step: int,
    large_step: int,
) -> pd.Series:
    """Assigns labels to timedelta series by comparing the given two series of timedelta objects.
    Args:
        timedelta_series (pd.Series): Original series.
        series_to_copmare (pd.Series): Timedelta series to compare with.
        first_timedelta_label (int): First label index of timedelta.
        small_step (int): Step size for the timedelta changes within 24 hrs.
        large_step (int): Step size for the timedelta changes over 24 hrs.
    Returns:
        final_labels (pd.Series): Series of labels. Missing labels are filled with 0.
    """
    # Define variables
    small_steps_per_day = 24 * 60 // small_step
    large_steps_per_day = 24 * 60 // large_step
    timedelta_sections = config.TD_SECTIONS
    # Calculate timedelta changes
    timedelta_diff = timedelta_series - series_to_copmare
    calendar_date_diff = (
        timedelta_series.dt.floor("D") - series_to_copmare.dt.floor("D")
    ).dt.days
    # Create masks
    end_of_large_step_span = timedelta_sections[-1][1]
    within_24hr_mask = timedelta_diff.dt.days < 1
    outside_span_mask = calendar_date_diff > end_of_large_step_span
    # Handle timedelta changes within 24 hours
    time_diff_within_24hr = timedelta_diff.mask(~within_24hr_mask, pd.NaT)
    labels_within_24hr = time_diff_within_24hr.dt.seconds // 60 // small_step
    labels_within_24hr += first_timedelta_label
    labels_within_24hr = labels_within_24hr.fillna(0)
    # Calculate changes in timedelta values over 24 hours
    minutes = timedelta_series.dt.seconds // 60
    calendar_date_diff = calendar_date_diff.mask(within_24hr_mask, None)
    calendar_date_diff = calendar_date_diff.mask(outside_span_mask, None)
    minute_diff = minutes.mask(within_24hr_mask, None)
    over24_df = pd.DataFrame(
        {config.COL_DAYS: calendar_date_diff, config.COL_MINUTES: minute_diff}
    )
    over24_df[["start_index", "start", "step"]] = 0
    section_first_index = first_timedelta_label + small_steps_per_day
    for section in timedelta_sections:
        start, end, step = section
        partitions_in_section = (end - start) // step + 1
        section_mask = (over24_df[config.COL_DAYS] >= start) & (
            over24_df[config.COL_DAYS] <= end
        )
        over24_df.loc[section_mask, "start_index"] = section_first_index
        over24_df.loc[section_mask, "start"] = start
        over24_df.loc[section_mask, "step"] = step
        # Update next first index
        section_first_index += partitions_in_section * large_steps_per_day
    labels_over_24hr = (
        (over24_df[config.COL_DAYS] - over24_df["start"]) // over24_df["step"]
    ) * large_steps_per_day + over24_df["start_index"]
    labels_over_24hr += over24_df[config.COL_MINUTES] // large_step
    labels_over_24hr = labels_over_24hr.fillna(0)

    # Handle timedelta values outside the large step span
    out_of_span_index = section_first_index
    labels_outside_span = (outside_span_mask * out_of_span_index).fillna(0).astype(int)

    # Concatenate labels
    final_labels = labels_within_24hr + labels_over_24hr + labels_outside_span

    return final_labels


def shuffle_timeline_matrix_indexes(
    timeline_matrix: torch.Tensor,
    pad_start: int = None,
    dsc_idx: int = None,
    eot_idx: int = None,
    lab_code_token_idx: int = None,
    k: int = 10,
    max_random_integer: int = 1000,
) -> list[list[int]]:
    """Creates series of randomly shuffled matrix indexes in a time-aware manner, where
    rows of timedelta are not shuffled and only rows within the same timedelta points are shuffled.
    By this random shuffling, discharge tokens ('[DSC]'), end-of-timeline tokens ('[EOT]') and demographic rows are not shuffled.
    Laboratory test items and related values are always shuffled in pairs.
    Args:
        timeline_matrix (torch.Tensor): Timeline matrix whose indexes are to be suffled.
        pad_start (int): Index where padding rows start in the matrix.
            If there is no padding rows in the matrix, this argument should be set to None. Default is None.
        dsc_idx (int): Embedding index of [DSC] token.
        eot_idx (int): Embedding index of [EOT] token.
        lab_code_token_idx (int): Embedding index of the token for laboratory test coding system ('[LAB]').
        *If these token index arguments are not explicitly passed, this function loads them on its own; however, they should be passed for the sake of efficiency.
        k (int): Number of suffled index sequences generated. Default is 10, therefore ten randomly shuffled index sequences are returned by default.
        max_random_integer (int): Maximum integer generated by np.random.randint function for the random shuffling.
            Default is 1000.
    Returns:
        shuffled_index_list (list): List of lists with each sublist is a series of shuffled index of the input matrix
    """
    # Initialize variables
    if dsc_idx is None:
        dsc_idx = load_special_token_index("[DSC]")
    if eot_idx is None:
        eot_idx = load_special_token_index("[EOT]")
    if lab_code_token_idx is None:
        lab_code_token_idx = load_special_token_index("[LAB]")
    demographic_rows = config.DEMOGRAPHIC_ROWS
    timestamp_dim = len(config.TIMEDELTA_COMPONENT_COLS)
    numeric_dim = config.NUMERIC_DIM
    categorical_start = timestamp_dim + numeric_dim
    matrix_length = timeline_matrix.size(0)
    # Ensure the matrix is on cpu and
    shuffled_matrix = timeline_matrix.cpu().detach()
    # Handle padding rows
    if pad_start is None:
        pad_start = matrix_length
        shuffled_length = matrix_length
        pad_indexes = []
    else:
        shuffled_matrix = shuffled_matrix[:pad_start]
        pad_indexes = list(range(pad_start, matrix_length, 1))
        shuffled_length = pad_start
    # Get specific indexes for masking
    demographic_idxs = np.arange(0, demographic_rows)
    timestamp_idxs = (
        (~torch.isnan(shuffled_matrix[:, 0:1])).nonzero(as_tuple=True)[0]
    ).numpy()
    discharge_idxs = (
        (
            shuffled_matrix[:, categorical_start : categorical_start + 1] == dsc_idx
        ).nonzero(as_tuple=True)[0]
    ).numpy()
    lab_idxs = (
        (
            shuffled_matrix[:, categorical_start : categorical_start + 1]
            == lab_code_token_idx
        ).nonzero(as_tuple=True)[0]
    ).numpy()
    eot_idx_list = (
        shuffled_matrix[:, categorical_start : categorical_start + 1] == eot_idx
    ).nonzero(as_tuple=True)[0]
    eot_idx = eot_idx_list[0] if len(eot_idx_list) > 0 else None
    # Create a template dataframe for shuffling
    shuffle_df_template = pd.DataFrame(
        {
            "idx": np.arange(0, shuffled_length),
        }
    )
    shuffle_df_template["ts"] = shuffle_df_template["idx"].copy()
    shuffle_df_template.loc[~shuffle_df_template.index.isin(timestamp_idxs), "ts"] = (
        None
    )
    shuffle_df_template["ts"] = shuffle_df_template["ts"].ffill().astype(int)
    # Handle the last row
    fix_bottom = False
    if eot_idx is not None:
        # Ensure that a row with a [EOT] token is excluded from shuffling
        fix_bottom = True
    elif (len(lab_idxs) > 0) and (lab_idxs[-1] == shuffled_length - 1):
        # Ensure that a row is excluded from shuffling if the very last row is a lab item without a result value
        lab_idxs = lab_idxs[:-1]
        fix_bottom = True

    # Shuffle indexes
    shuffled_index_list = []
    for _ in range(k):

        # Create a new dataframe for shuffling
        shuffle_df = shuffle_df_template.copy()

        # Giving each row a random number
        shuffle_df["priority"] = np.random.randint(
            1, max_random_integer + 1, shuffled_length
        )
        # Place timestamps at the top of each timestamp segment
        shuffle_df.loc[timestamp_idxs, "priority"] = 0
        # Prevent shuffling rows of demographic data
        shuffle_df.loc[demographic_idxs, "priority"] = 0
        # Prevent shuffling discharge tokens
        # NOTE: A token for disposition may or may not follow. But They should remain in the same position.
        shuffle_df.loc[discharge_idxs, "priority"] = 0
        # Pair lab results with lab items
        labs = shuffle_df.loc[lab_idxs, "priority"]
        labs.index += 1  # <- Shift index by one for corresponding lab values
        shuffle_df.loc[lab_idxs + 1, "priority"] = labs.values
        # Ensure that the last row remains at the last.
        if fix_bottom:
            # To prevent the last lab item row to be placed elsewhere without a result value.
            shuffle_df.loc[shuffled_length - 1, "priority"] = max_random_integer + 1
        # Shuffle indexes by sorting values by the newly assigned 'priority' column
        shuffle_df = shuffle_df.sort_values(
            ["ts", "priority", "idx"], ascending=[True, True, True]
        )
        # Create a series of shuffled indexes
        idxs = shuffle_df.loc[:, "idx"].tolist()
        # Append indexes for padding
        idxs += pad_indexes
        # Append the result to the final list
        shuffled_index_list.append(idxs)

    return shuffled_index_list
