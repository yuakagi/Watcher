"""Module to create a dictionary to map categorical values to tokens and embedding indexes."""

import re
import json
import glob
import pandas as pd
from .....general_params import watcher_config as config
from .....general_params import get_settings
from .....utils import load_nonnumeric_stats


def create_tokenization_map():
    """Creates dictionary to map all categorical values to tokens,
        and assigns all tokens including special tokens unique embedding indexes.

    This function creates and saves the two different dictionaries ,'token_reference' and
    'tokenization_map'. The former contains a list of unique tokens and their corresponding
    embedding indexes. It also contains some other references. The latter is used for mapping codes and
    tokens to series of indexes for embeddings.

        - 'token_reference':
            This dictionary contains reference data for tokenization.
            This has the following sections:
                - 'vocabulary size' : Total vocabulary size of the token embedding layer
                - 'categorical_dim': Maximal number indexes that can be assigned to a single code.
                - 'tokens and indexes':
                    - 'special tokens':
                        - 'token': List of special tokens
                        - 'index': List of token indexes
                    - 'diagnosis_codes':
                        - 'token': List of tokens for diagnosis codes
                        - 'index': List of token indexes
                    - 'medication_code':
                        - 'token': List of tokens for medication codes
                        - 'index': List of token indexes
                    - 'lab_test_code':
                        - 'token': List of tokens for lab test codes
                        - 'index': List of token indexes
                        ('token' and 'index' are paired)

        - tokenization_map:
            This dictionary contain lists of codes and tokens. This is designed so that they can be loaded as a pd.DataFrame to
            ease tokenization processes. By merging tables using on 'config.COL_ORIGINAL_VALUE' column, codes are mapped to sequences of
            embedding indexes.
            The standardized codes (diagnosis, lab test and medication) are sorted lexicographically.
            This has the following sections in this order:
                - 'special tokens'
                    - 'config.COL_ORIGINAL_VALUE': List of special tokens
                    - 'tokenized_code': List of tokenized values. This list is exactly same as one in
                        'config.COL_ORIGINAL_VALUE', but this section is made for consistency with other
                        parent sections.
                    - 'c*' columns : each column contains an index to represent the original value.
                - 'diagnosis_codes':
                    - 'config.COL_ORIGINAL_VALUE': List of unique diagnosis codes
                    - 'tokenized_code': List of tokenized diagnosis codes
                    - 'c*' columns : ...
                - 'lab_test_code':
                    - 'config.COL_ORIGINAL_VALUE': List of unique lab test codes
                    - 'tokenized_code': List of tokenized lab test codes
                    - 'c*' columns : ...
                - 'medication_codes':
                    - 'config.COL_ORIGINAL_VALUE': List of unique mecication  codes
                    - 'tokenized_code': List of tokenized medication codes
                    - 'c*' columns : ...

    The parameter 'vocabulary size' and 'categorical_dim' are Watcher model's hyperparameters.

    Args:
        None (This function reads configurations and file paths from settings.)
    """
    # Initialize
    tokenization_map = {}
    vocab_size = 0
    token_reference = {
        "vocabulary_size": 0,
        "tokens_and_indexes": {},
    }
    # Define tokenization params
    categorical_dim = get_settings("CATEGORICAL_DIM")
    code_tokenization_params = {
        config.DX_CODE: {
            "coding_system_token": "[DX]",
            "unique_count_table": get_settings("DX_CODE_COUNTS_PTH"),
            "segment_sizes": get_settings("DX_CODE_SEGMENTS"),
        },
        config.MED_CODE: {
            "coding_system_token": "[MED]",
            "unique_count_table": get_settings("MED_CODE_COUNTS_PTH"),
            "segment_sizes": get_settings("MED_CODE_SEGMENTS"),
        },
        config.LAB_CODE: {
            "coding_system_token": "[LAB]",
            "unique_count_table": None,
            "segment_sizes": get_settings("LAB_CODE_SEGMENTS"),
        },
    }

    # Collect unique codes from the training data
    total_unique_code_dict = {}
    for code_system in [config.DX_CODE, config.MED_CODE]:
        train_codes = []
        path_pattern = code_tokenization_params[code_system]["unique_count_table"]
        for file in glob.glob(path_pattern.replace(".csv", "*.csv")):
            temp_df = pd.read_csv(
                file,
                header=0,
                na_values=config.NA_VALUES,
                usecols=[config.COL_ORIGINAL_VALUE, "train_ID_and_train_period"],
                dtype={
                    config.COL_ORIGINAL_VALUE: str,
                    "train_ID_and_train_period": int,
                },
            )
            train_mask = temp_df["train_ID_and_train_period"] >= 1
            temp_df = temp_df.loc[train_mask, [config.COL_ORIGINAL_VALUE]]
            codes = list(temp_df[config.COL_ORIGINAL_VALUE].unique())
            train_codes += codes
        train_codes = list(set(train_codes))
        total_unique_code_dict[code_system] = train_codes

    # laboratory test codes are collected from the laboratory test result stats.
    lab_codes = []
    for file in [
        get_settings("NUMERIC_PERCENTILES_PTN").replace("*", "train"),
        get_settings("NONNUMERIC_STATS_PTN").replace("*", "train"),
    ]:
        temp_df = pd.read_csv(
            file, header=0, na_values=config.NA_VALUES, usecols=[config.COL_ITEM_CODE]
        )
        codes = list(temp_df[config.COL_ITEM_CODE].unique())
        lab_codes += codes
    total_unique_code_dict[config.LAB_CODE] = list(set(lab_codes))

    # *****************
    # * Special Tokens *
    # *****************

    # Load special tokens from different sources
    discharge_related_tokens = config.DISCHARGE_STATUS_TOKENS.values()
    sex_related_tokens = config.SEX_TOKENS.values()
    nonnumeric_stats = load_nonnumeric_stats()
    nonnumeric_tokens = nonnumeric_stats[config.COL_TOKEN].unique().tolist()

    # Sort tokens
    # (Simply applying 'sort()' could lead to errors.)
    rank_rx = r"\[top(\d+)\]"
    top_n_ranks = []
    other_nonnumeric_tokens = []
    for t in nonnumeric_tokens:
        group = re.search(rank_rx, t)
        if group:
            rank = int(group[1])
            top_n_ranks.append(rank)
        else:
            other_nonnumeric_tokens.append(t)
    sorted_top_n_tokens = [
        f"[top{n}]" for n in range(min(top_n_ranks), max(top_n_ranks) + 1, 1)
    ]
    nonnumeric_tokens = other_nonnumeric_tokens + sorted_top_n_tokens

    # Aggregate special tokens
    special_tokens = config.SPECIAL_TOKENS.copy()
    special_tokens += discharge_related_tokens  # <- discharge dispositions
    special_tokens += sex_related_tokens  # <- patient sex tokens
    special_tokens += nonnumeric_tokens  # <- lab nonnumeric value tokens

    # Assign unique indexes to special tokens
    special_token_indexes = list(range(len(special_tokens)))

    # Write to the dictionary
    token_reference["tokens_and_indexes"]["special_tokens"] = {
        config.COL_TOKEN: special_tokens,
        "index": special_token_indexes,
    }
    special_token_map = pd.DataFrame(
        {
            config.COL_ORIGINAL_VALUE: special_tokens,
            config.COL_TOKENIZED_VALUE: special_tokens,
            "c0": special_token_indexes,
        }
    )
    tokenization_map["special_tokens"] = special_token_map

    # Count up vocab size
    n_special_tokens = len(special_tokens)
    vocab_size += n_special_tokens

    # **********************
    # * Standardized codes *
    # **********************
    # Extract unique tokens and give unique indexes to them
    for coding_system_name, params in code_tokenization_params.items():
        # Set params
        unique_codes = total_unique_code_dict[coding_system_name]
        coding_system_token = params["coding_system_token"]
        system_token_index = special_tokens.index(coding_system_token)
        segment_sizes = params["segment_sizes"]

        # Initialize final product dataframe
        code_map_df = pd.DataFrame({config.COL_ORIGINAL_VALUE: unique_codes})
        code_map_df[config.COL_TOKENIZED_VALUE] = coding_system_token
        code_map_df["c0"] = system_token_index
        if coding_system_name == config.DX_CODE:
            # Handle provisional flags in diagnosis codes
            prv_flag = code_map_df[config.COL_ORIGINAL_VALUE].str.endswith(
                config.PROV_SUFFIX, na=False
            )
            code_map_df[config.COL_PROVISIONAL_FLAG] = prv_flag.astype(int)
            code_map_df[config.COL_ORIGINAL_VALUE] = code_map_df[
                config.COL_ORIGINAL_VALUE
            ].str.replace(config.PROV_SUFFIX, "", regex=False)

        # Slice out codes into segments and tokenize
        placeholders = ""
        token_dict = {config.COL_TOKEN: [], "index": []}
        if isinstance(segment_sizes, list) and segment_sizes:
            seg_start, seg_end = 0, 0
            for segment_no, seg_size_str in enumerate(segment_sizes, start=1):
                # Tokenize codes by slicing out code segments from codes
                seg_size = int(seg_size_str)
                seg_end += seg_size
                tokens = code_map_df[config.COL_ORIGINAL_VALUE].str.slice(
                    seg_start, seg_end
                )
                missing_token_mask = tokens == ""
                tokens[~missing_token_mask] = placeholders + tokens[~missing_token_mask]
                tokens = tokens.fillna("")
                code_map_df[config.COL_TOKEN] = tokens

                # Assign unique indexes to each token
                unique_tokens = sorted(list(set(tokens.to_list())))
                if "" in unique_tokens:
                    # Remove empty strings
                    unique_tokens.remove("")
                unique_token_indexes = list(
                    range(vocab_size, vocab_size + len(unique_tokens))
                )
                token_dict[config.COL_TOKEN] += unique_tokens
                token_dict["index"] += unique_token_indexes

                # Map tokens in the original codes
                token_map_df = pd.DataFrame(token_dict)
                code_map_df = code_map_df.merge(
                    token_map_df, on=config.COL_TOKEN, how="left"
                )
                code_map_df["index"] = code_map_df["index"].fillna(0).astype(int)
                code_map_df = code_map_df.rename(columns={"index": f"c{segment_no}"})

                # Append sliced out tokens to the final product token col
                code_map_df[config.COL_TOKENIZED_VALUE] += (
                    "," + code_map_df[config.COL_TOKEN]
                )

                # Drop temporary col
                code_map_df = code_map_df.drop([config.COL_TOKEN], axis=1)

                # Adjust placeholders
                placeholders += "*" * (seg_end - seg_start)

                # Count up
                seg_start += seg_size
                vocab_size += len(unique_tokens)

        # Treat a unique code as a single token
        else:
            unique_token_indexes = list(
                range(vocab_size, vocab_size + len(unique_codes))
            )
            code_map_df[config.COL_TOKENIZED_VALUE] = (
                coding_system_token + "," + code_map_df[config.COL_ORIGINAL_VALUE]
            )
            code_map_df["c1"] = pd.Series(unique_token_indexes)
            token_dict[config.COL_TOKEN] += unique_codes
            token_dict["index"] += unique_token_indexes
            vocab_size += +len(unique_codes)

        # Add data to the final products
        token_reference["tokens_and_indexes"][coding_system_name] = token_dict
        code_map_df[config.COL_TOKENIZED_VALUE] = code_map_df[
            config.COL_TOKENIZED_VALUE
        ].str.rstrip(",")
        code_map_df = code_map_df.sort_values(config.COL_ORIGINAL_VALUE, ascending=True)
        tokenization_map[coding_system_name] = code_map_df

    # Save the vocabulary size for later use.
    token_reference["vocabulary_size"] = vocab_size
    token_reference["categorical_dim"] = categorical_dim

    # Pad 'c*' columns
    embedding_index_cols = [f"c{i}" for i in range(categorical_dim)]
    for k, v in tokenization_map.items():
        columns = v.columns
        for col in embedding_index_cols:
            if col not in columns:
                v[col] = 0
        tokenization_map[k] = v

    # Handle '[PRV]' token for diagnosis codes lastly
    prv_index = special_tokens.index("[PRV]")
    dx_code_map = tokenization_map[config.DX_CODE]
    prv_mask = dx_code_map[config.COL_PROVISIONAL_FLAG] == 1
    dx_code_map.loc[prv_mask, config.COL_ORIGINAL_VALUE] += config.PROV_SUFFIX
    dx_code_map.loc[prv_mask, config.COL_TOKENIZED_VALUE] += ",[PRV]"
    for col in embedding_index_cols:
        updated_prv_mask = dx_code_map[config.COL_PROVISIONAL_FLAG] == 1
        zero_mask = dx_code_map[col] == 0
        prv_index_added_mask = updated_prv_mask & zero_mask
        dx_code_map.loc[prv_index_added_mask, col] = prv_index
        dx_code_map[config.COL_PROVISIONAL_FLAG] = dx_code_map[
            config.COL_PROVISIONAL_FLAG
        ].mask(prv_index_added_mask, 0)
    dx_code_map = dx_code_map.drop(config.COL_PROVISIONAL_FLAG, axis=1)
    dx_code_map = dx_code_map.sort_values(config.COL_ORIGINAL_VALUE, ascending=True)
    tokenization_map[config.DX_CODE] = dx_code_map

    # Finalize tokenization_map
    for k, v in tokenization_map.items():
        tokenization_map[k] = v.to_dict(orient="list")

    # Save the references.
    with open(get_settings("TOKENIZATION_MAP_PTH"), "w", encoding="utf-8") as f:
        json.dump(tokenization_map, f, indent=2)
    with open(get_settings("TOKEN_REFERENCE_PTH"), "w", encoding="utf-8") as f:
        json.dump(token_reference, f, indent=2)
