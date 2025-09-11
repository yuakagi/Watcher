"""Module to create catalogs"""

import json
import numpy as np
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import (
    compute_numeric_steps,
    discretize_percentiles,
    load_categorical_dim,
    map_special_tokens_to_name,
    map_code_to_name,
)


def create_catalog():
    """
    Creates catalogs for preprocessing, labeling, and inference, and saves them as CSV files.

    The catalog tables are indexed by inference vocabulary indexes, used as labels during model training.
    The catalog contains the following information:

    - 'index': Inference vocabulary indexes in the order of special tokens (including nonnumeric lab values), diagnosis codes, medication codes, medication codes, and numeric bins.
                Standardized codes are sorted lexicographically, and numeric values are in ascending order.
    - 'config.COL_ORIGINAL_VALUE': Original values.
    - 'tokenized_code': Tokenized codes, separated into individual components. Special tokens retain their original values.
    - 'c*' columns: Embedding indexes used by the model's categorical embedding layer.
    - 'text': Plain text representation of each entity.
    - 'config.COL_LABEL': Label indexes used for training.

    A full catalog and its components (e.g., special tokens, diagnosis codes, medication codes) are saved separately for convenience.
    """

    # Load references
    with open(get_settings("TOKENIZATION_MAP_PTH"), "r", encoding="utf-8") as f:
        tokenization_map = json.load(f)
    categorical_dim = load_categorical_dim()
    categorical_columns = [f"c{i}" for i in range(categorical_dim)]
    final_columns = (
        [config.COL_ORIGINAL_VALUE, config.COL_TEXT, config.COL_TOKENIZED_VALUE]
        + categorical_columns
        + [config.COL_LABEL]
    )

    # *****************
    # * Special tokens *
    # *****************
    special_token_map = tokenization_map["special_tokens"]
    special_token_catalog = pd.DataFrame(special_token_map)
    special_token_catalog[config.COL_TEXT] = map_special_tokens_to_name(
        special_token_catalog[config.COL_ORIGINAL_VALUE]
    )
    discharge_related_tokens = config.DISCHARGE_STATUS_TOKENS.values()
    sex_related_tokens = config.SEX_TOKENS.values()
    operational_special_tokens = config.SPECIAL_TOKENS

    # Count records
    n_special_tokens = len(special_token_catalog)
    # Place labels
    special_token_catalog[config.COL_LABEL] = special_token_catalog.index
    # Sort columns
    special_token_catalog = special_token_catalog[final_columns]

    # ****************
    # * diagnosis codes *
    # ****************
    # Create diagnosis code table
    dx_code_map = tokenization_map[config.DX_CODE]
    dx_code_catalog = pd.DataFrame(dx_code_map)
    dx_code_catalog[config.COL_TEXT] = map_code_to_name(
        dx_code_catalog[config.COL_ORIGINAL_VALUE].str.replace(
            config.PROV_SUFFIX, "", regex=False
        ),
        code_type=config.DX_CODE,
    )
    prv_mask = dx_code_catalog[config.COL_ORIGINAL_VALUE].str.endswith(
        config.PROV_SUFFIX, na=False
    )
    dx_code_catalog.loc[prv_mask, config.COL_TEXT] = (
        dx_code_catalog.loc[prv_mask, config.COL_TEXT] + " " + config.PROV_SUFFIX
    )

    # Count records
    n_dx_code = len(dx_code_catalog)

    # Place labels
    dx_code_index_start = n_special_tokens
    dx_code_index_end = n_special_tokens + n_dx_code
    dx_code_catalog[config.COL_LABEL] = list(
        range(dx_code_index_start, dx_code_index_end)
    )

    # Sort columns
    dx_code_catalog = dx_code_catalog[final_columns]

    # *************
    # * medication codes *
    # *************
    med_token_map = tokenization_map[config.MED_CODE]
    med_code_catalog = pd.DataFrame(med_token_map)
    med_code_catalog[config.COL_TEXT] = map_code_to_name(
        med_code_catalog[config.COL_ORIGINAL_VALUE], code_type=config.MED_CODE
    )

    # Count records
    n_med_code = len(med_code_catalog)

    # Place labels
    med_code_index_start = dx_code_index_end
    med_code_index_end = med_code_index_start + n_med_code
    med_code_catalog[config.COL_LABEL] = list(
        range(med_code_index_start, med_code_index_end)
    )

    # Sort columns
    med_code_catalog = med_code_catalog[final_columns]

    # ****************
    # * mecication  codes *
    # ****************
    lab_token_map = tokenization_map[config.LAB_CODE]
    lab_code_catalog = pd.DataFrame(lab_token_map)
    lab_code_catalog[config.COL_TEXT] = map_code_to_name(
        lab_code_catalog[config.COL_ORIGINAL_VALUE], code_type=config.LAB_CODE
    )

    # Count records
    n_lab_code = len(lab_code_catalog)

    # Place labels
    lab_code_index_start = med_code_index_end
    lab_code_index_end = lab_code_index_start + n_lab_code
    lab_code_catalog[config.COL_LABEL] = list(
        range(lab_code_index_start, lab_code_index_end)
    )

    # Sort columns
    lab_code_catalog = lab_code_catalog[final_columns]

    # ****************************************
    # * Concatenate categorical value tables *
    # ****************************************
    categorical_catalog = pd.concat(
        [
            special_token_catalog,
            dx_code_catalog,
            med_code_catalog,
            lab_code_catalog,
        ]
    )

    # ******************
    # * Numeric values *
    # ******************
    n_numeric_bins = get_settings("NUMERIC_BINS")

    # <-- percentile -->
    numeric_steps = compute_numeric_steps(num_bins=n_numeric_bins)
    precentile_steps = discretize_percentiles(num_bins=n_numeric_bins)
    original_nums = numeric_steps.astype(str)
    nums_str = pd.Series((precentile_steps * 100).astype(str))
    nums_str += " percentile"
    nums_str = nums_str.values

    # Set up catalog
    numeric_catalog = pd.DataFrame({config.COL_ORIGINAL_VALUE: original_nums})
    numeric_catalog[config.COL_TEXT] = nums_str
    # Pad cols
    numeric_catalog[config.COL_TOKENIZED_VALUE] = ""
    for col in categorical_columns:
        numeric_catalog[col] = 0
    # Place labels
    numeric_index_start = lab_code_index_end
    numeric_index_end = numeric_index_start + len(original_nums)
    numeric_catalog[config.COL_LABEL] = range(numeric_index_start, numeric_index_end)
    # Sort columns
    numeric_catalog = numeric_catalog[final_columns]

    # ********************
    # * Timedelta values *
    # ********************
    # For small steps
    step_texts = []
    td_small_step = get_settings("TD_SMALL_STEP")
    n_small_dates = list(range(0, config.TD_SECTIONS[0][0]))
    n_small_steps_a_day = int(24 * 60 // td_small_step)
    for dates in n_small_dates:
        for i in range(0, n_small_steps_a_day):
            lower_end_total_minutes = td_small_step * i
            upper_end_total_minutes = (
                lower_end_total_minutes + td_small_step
            ) - 1  # <- -1 for closed interval end

            td_small_text = (
                f"+ {lower_end_total_minutes}-{upper_end_total_minutes} minutes"
            )

            step_texts.append(td_small_text)

    # For large steps
    large_text_templates = []
    td_large_step = get_settings("TD_LARGE_STEP")
    n_large_steps_a_day = int(24 * 60 // td_large_step)
    for i in range(0, n_large_steps_a_day):
        lower_end_total_minutes = td_large_step * i
        upper_end_total_minutes = (
            lower_end_total_minutes + td_large_step - 1
        )  # <- -1 for closed end
        lower_end_hr = str(lower_end_total_minutes // 60).zfill(2)
        upper_end_hr = str(upper_end_total_minutes // 60).zfill(2)
        lower_end_min = str(lower_end_total_minutes % 60).zfill(2)
        upper_end_min = str(upper_end_total_minutes % 60).zfill(2)
        text = f" & {lower_end_hr}:{lower_end_min}-{upper_end_hr}:{upper_end_min}"
        large_text_templates.append(text)

    for section in config.TD_SECTIONS:
        min_dates, max_dates, step_size = section
        if step_size == 1:
            for dates in range(min_dates, max_dates + 1):
                for template_text in large_text_templates:
                    td_large_text = f"+ {dates} days" + template_text
                    step_texts.append(td_large_text)
        else:
            for dates in range(min_dates, max_dates, step_size):
                for template_text in large_text_templates:
                    td_large_text = (
                        f"+ {dates}-{dates+step_size-1} days" + template_text
                    )
                    step_texts.append(td_large_text)

    # For timestamps over large step span
    max_large_step_days = config.TD_SECTIONS[-1][1]
    out_of_span_text = f"over {max_large_step_days} days"
    step_texts.append(out_of_span_text)

    # Initialize dataframe
    n_timedelta = len(step_texts)
    timedelta_catalog = pd.DataFrame(
        {config.COL_ORIGINAL_VALUE: np.full(n_timedelta, "")}
    )
    timedelta_catalog[config.COL_TOKENIZED_VALUE] = ""
    timedelta_catalog[config.COL_TEXT] = step_texts
    # timedelta_catalog["embedding_indexes"] = [[] for _ in range(n_timedelta)]

    # Pad categorical cols
    for col in categorical_columns:
        timedelta_catalog[col] = 0

    # Place labels
    timedelta_index_start = numeric_index_end
    timedelta_index_end = timedelta_index_start + n_timedelta
    timedelta_catalog[config.COL_LABEL] = list(
        range(timedelta_index_start, timedelta_index_end)
    )

    # Sort columns
    timedelta_catalog = timedelta_catalog[final_columns]

    # **************************
    # * Concatenate all tables *
    # **************************
    full_catalog = pd.concat([categorical_catalog, numeric_catalog, timedelta_catalog])
    full_catalog = full_catalog.reset_index(drop=True)

    # Save the full catalog
    full_catalog.to_csv(get_settings("CATALOG_FILE_PTH"), header=True, index=False)

    # ******************************
    # * Summarise the catalog info *
    # ******************************
    # Collect indexes
    special_token_indexes = special_token_catalog[config.COL_LABEL].tolist()
    operational_token_indexes = special_token_catalog[
        special_token_catalog[config.COL_ORIGINAL_VALUE].isin(
            operational_special_tokens
        )
    ].index.to_list()
    sex_token_indexes = special_token_catalog[
        special_token_catalog[config.COL_ORIGINAL_VALUE].isin(sex_related_tokens)
    ].index.to_list()
    discharge_related_token_indexes = special_token_catalog[
        special_token_catalog[config.COL_ORIGINAL_VALUE].isin(discharge_related_tokens)
    ].index.to_list()
    nonnum_indexes = list(
        set(special_token_indexes)
        - set(operational_token_indexes)
        - set(sex_token_indexes)
        - set(discharge_related_token_indexes)
    )
    numeric_indexes = numeric_catalog[config.COL_LABEL].tolist()
    timedelta_indexes = timedelta_catalog[config.COL_LABEL].tolist()
    dx_code_indexes = dx_code_catalog[config.COL_LABEL].tolist()
    med_code_indexes = med_code_catalog[config.COL_LABEL].tolist()
    lab_code_indexes = lab_code_catalog[config.COL_LABEL].tolist()
    all_code_indexes = dx_code_indexes + med_code_indexes + lab_code_indexes
    all_categorical_indexes = (
        special_token_indexes + dx_code_indexes + med_code_indexes + lab_code_indexes
    )
    # Create a dict to store the info
    catalog_info = {}
    catalog_info["all_indexes"] = {
        "full": full_catalog.index.to_list(),
        "all_special_tokens": special_token_indexes,
        "sex_tokens": sex_token_indexes,
        "discharge_disposition_tokens": discharge_related_token_indexes,
        "operational_special_tokens": operational_token_indexes,
        "nonnumeric_lab_values": nonnum_indexes,
        "numeric_lab_values": numeric_indexes,
        "timedelta": timedelta_indexes,
        config.DX_CODE: dx_code_indexes,
        config.MED_CODE: med_code_indexes,
        config.LAB_CODE: lab_code_indexes,
        "all_codes": all_code_indexes,
        "all_categorical": all_categorical_indexes,
    }
    catalog_info["max_indexes"] = {
        k: max(v) for k, v in catalog_info["all_indexes"].items()
    }
    catalog_info["min_indexes"] = {
        k: min(v) for k, v in catalog_info["all_indexes"].items()
    }
    # Save the dict
    with open(get_settings("CATALOG_INFO_PTH"), "w", encoding="utf-8") as f:
        json.dump(catalog_info, f, indent=2)
