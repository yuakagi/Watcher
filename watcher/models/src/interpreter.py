"""Module for interpreting Watcher's outputs"""

import os
import numpy as np
import pandas as pd
import torch
from ...general_params import watcher_config as config
from ...utils import (
    load_catalog,
    load_catalog_info,
    load_lab_percentiles,
    load_nonnumeric_stats,
    load_numeric_stats,
    discretize_percentiles,
    compute_numeric_steps,
    timedelta_to_text,
    extract_timedelta_sequence,
    format_df,
)


class WatcherInterpreter(object):
    """An object that holds values and functions necessary for interpretation of outputs from Watcher.

    Watcher itself has a WatcherInterpreter object inside, and uses this for inference.
    """

    def _load_all_catalogs(self, catalogs_dir: str) -> dict:
        """Loads the catalogs of all types"""
        catalogs = {}
        for catalog_type in [
            "full",
            "all_special_tokens",
            "sex_tokens",
            "discharge_disposition_tokens",
            "operational_special_tokens",
            "nonnumeric_lab_values",
            "numeric_lab_values",
            "timedelta",
            config.DX_CODE,
            config.MED_CODE,
            config.LAB_CODE,
            "all_codes",
            "all_categorical",
        ]:
            # Load a catalog
            temp_catalog = load_catalog(catalog_type, catalogs_dir=catalogs_dir)
            # Save the catalog to the catalog dictionary
            catalogs[catalog_type] = temp_catalog

        return catalogs

    def _create_special_token_dict(self) -> dict:
        """Creates a dictionary of special token indexes."""
        df = self.catalogs["all_special_tokens"]
        special_token_dict = {}
        # Create a dictionary of token embedding indexes
        for _, row in df[[config.COL_TOKENIZED_VALUE, config.COL_LABEL]].iterrows():
            token, token_index = row
            special_token_dict[token] = token_index
        return special_token_dict

    def __init__(
        self, n_numeric_bins: int, catalogs_dir: str, lab_stats_dir: str | None = None
    ):
        # Attributes from config
        self.n_demographic_rows = config.DEMOGRAPHIC_ROWS
        self.timedelta_dim = len(config.TIMEDELTA_COMPONENT_COLS)
        self.numeric_dim = config.NUMERIC_DIM
        self.ignored_index = config.LOGITS_IGNORE_INDEX
        # Load catalogs and references for indexes
        self.catalogs = self._load_all_catalogs(catalogs_dir)
        # Assign catalog info to the attributes
        catalog_info = load_catalog_info(catalogs_dir=catalogs_dir)
        self.catalog_index_lists = catalog_info["all_indexes"]
        self.min_indexes = catalog_info["min_indexes"]
        self.max_indexes = catalog_info["max_indexes"]
        # Create a dictionary for special token indexes
        self.special_token_dict = self._create_special_token_dict()
        # Create a table for categorical laboratory test value inference
        if lab_stats_dir is not None:
            percentiles_path = os.path.join(
                lab_stats_dir, config.LAB_PERCENTILES_PATTERN
            ).replace("*", config.TRAIN)
            nonnum_stats_path = os.path.join(
                lab_stats_dir, config.LAB_NONNUM_STATS_PATTERN
            ).replace("*", config.TRAIN)
            num_stats_path = os.path.join(
                lab_stats_dir, config.LAB_NUM_STATS_PATTERN
            ).replace("*", config.TRAIN)
        else:
            nonnum_stats_path = percentiles_path = num_stats_path = None
        nonnum_stats = load_nonnumeric_stats(file_path=nonnum_stats_path)
        self.nonnum_stats = nonnum_stats
        self.nonnumeric_result_table = self.nonnum_stats[
            [config.COL_ITEM_CODE, config.COL_TOKEN, config.COL_NONNUMERIC]
        ]
        # Create a table for numeric laboratory test value inference
        # NOTE: For units, most frequently encountered units are selected for each medication code (single_unit=True).
        # <-- percentiles -->
        self.percentiles, self.percentile_cols = load_lab_percentiles(
            single_unit=True, file_path=percentiles_path
        )
        self.numeric_steps = compute_numeric_steps(num_bins=n_numeric_bins)
        self.percentile_steps = discretize_percentiles(num_bins=n_numeric_bins)
        # <-- z-score -->
        self.num_stats = load_numeric_stats(single_unit=True, file_path=num_stats_path)
        # Stats for preprocessing
        # NOTE: This may be redundant.
        self.percentiles_for_preproc, _ = load_lab_percentiles(
            single_unit=False, file_path=percentiles_path
        )
        self.num_stats_for_preproc = load_numeric_stats(
            single_unit=False, file_path=num_stats_path
        )

    def translate_numerics(self, lab_codes: np.ndarray, numerics: np.ndarray):
        """Compute percentile values back to the original scale.

        Args:
            lab_codes (np.ndarray): Series of laboratory test codes.
            numerics (np.ndarray): Series of numeric values (in percentiles).
                Length of the two arrays must be the same.
        Returns:
            translated_numerics (np.ndarray): Series of translated numeric values (string).
                The length of this series matches the original series. Untranslated values are replaced with empty strings.
        """
        # Assertion
        assert len(lab_codes) == len(
            numerics
        ), "The length of `lab_codes` and `numerics` must be the same"
        # Set up a dataframe
        num_df = pd.DataFrame(
            {config.COL_ITEM_CODE: lab_codes, config.COL_NUMERIC: numerics}
        )
        # <--- Strategy using percentile --->
        num_df = pd.merge(num_df, self.percentiles, on=config.COL_ITEM_CODE, how="left")
        mask = num_df[self.percentile_cols + [config.COL_NUMERIC]].notna().all(axis=1)
        num_series = num_df.loc[mask, config.COL_NUMERIC].values
        unscaled_vals = num_df.loc[mask, self.percentile_cols].values
        # Get the nearest percentile indexes
        num_indexes = np.argmin(
            np.abs(self.numeric_steps.reshape(-1, 1) - num_series), axis=0
        )
        # Map percentile to actual value
        target_vals = unscaled_vals[np.arange(0, len(num_indexes)), num_indexes]

        # Clean the values
        target_vals = np.round(target_vals, decimals=5).astype(str)
        target_vals = np.char.strip(target_vals)
        # Assign the cleand values
        num_df[config.COL_TEXT] = ""
        num_df.loc[mask, config.COL_TEXT] = target_vals
        num_df.loc[mask, config.COL_TEXT] += " "
        num_df.loc[mask, config.COL_TEXT] += num_df.loc[mask, "unit"].str.strip()
        # Finalize
        translated_numerics = num_df[config.COL_TEXT].values

        return translated_numerics

    def _interpret_timeline(
        self,
        catalog_indexes: list,
        numeric_sequence: list,
        timedelta_sequence: list,
        lengths: list[int] = None,
        readable_timedelta: bool = False,
    ) -> tuple:
        """Helper function for 'interpret_timeline'."""
        # Initialize dataframe
        df = pd.DataFrame(
            {
                "catalog_index": catalog_indexes,
                config.COL_NUMERIC: numeric_sequence,
                config.COL_TIMEDELTA: timedelta_sequence,
            }
        )
        # ********************************
        # * Interpret categorical values *
        # ********************************
        # Prepare a mapping catalog
        categorical_map = self.catalogs["all_categorical"]
        categorical_map = categorical_map[
            [config.COL_LABEL, config.COL_ORIGINAL_VALUE, config.COL_TEXT]
        ]
        # Add text column by merging
        df = pd.merge(
            df,
            categorical_map,
            left_on="catalog_index",
            right_on=config.COL_LABEL,
            how="left",
        )
        # Handle nonnumeric laboratory test values
        df["shifted_code"] = df[config.COL_ORIGINAL_VALUE].shift(1)
        df = pd.merge(
            df,
            self.nonnumeric_result_table,
            left_on=["shifted_code", config.COL_ORIGINAL_VALUE],
            right_on=[config.COL_ITEM_CODE, config.COL_TOKEN],
            how="left",
        )

        mapped_nonnum_mask = ~(df[config.COL_NONNUMERIC].isna())
        df[config.COL_TEXT] = df[config.COL_TEXT].mask(
            mapped_nonnum_mask, df[config.COL_NONNUMERIC]
        )
        # Drop unnecessary columns
        df = df[
            [
                "catalog_index",
                config.COL_NUMERIC,
                config.COL_TIMEDELTA,
                "shifted_code",
                config.COL_TEXT,
                config.COL_ORIGINAL_VALUE,
            ]
        ]

        # ****************************
        # * Interpret numeric values *
        # ****************************
        # Add units, and percentiles
        numeric_mask = df[config.COL_NUMERIC].notna()
        translated_nums = self.translate_numerics(
            lab_codes=df.loc[numeric_mask, "shifted_code"].values,
            numerics=df.loc[numeric_mask, config.COL_NUMERIC].values,
        )
        df.loc[numeric_mask, config.COL_TEXT] = translated_nums

        # *******************************
        # * Interpret timedelta values *
        # *******************************
        timedelta_mask = df["catalog_index"] > self.max_indexes["numeric_lab_values"]
        timedelta_mask[self.n_demographic_rows] = True
        if lengths is not None:
            timeline_starts = [0] + np.cumsum(lengths).tolist()
            timeline_starts = timeline_starts[:-1]
            is_initial_row = (
                df["catalog_index"] == config.LOGITS_IGNORE_INDEX
            ) & df.index.isin(timeline_starts)
            timedelta_mask = timedelta_mask | is_initial_row
        else:
            if df.loc[0, "catalog_index"] == config.LOGITS_IGNORE_INDEX:
                timedelta_mask[0] = True
        td_series = df.loc[timedelta_mask, config.COL_TIMEDELTA]
        if readable_timedelta:
            timedelta_texts = timedelta_to_text(td_series)
        else:
            timedelta_texts = td_series.astype(str)
        df.loc[timedelta_mask, config.COL_TEXT] = timedelta_texts.values

        # ************
        # * Finalize *
        # ************
        # Create the product lists
        interpreted_texts = df[config.COL_TEXT].fillna("").to_list()
        interpreted_codes = df[config.COL_ORIGINAL_VALUE].fillna("").to_list()
        timedelta_indexes = df[timedelta_mask].index.to_list()

        return interpreted_texts, interpreted_codes, timedelta_indexes

    def interpret_timeline(
        self,
        timeline_list: torch.Tensor | list[torch.Tensor],
        catalog_idx_list: list[list[int]] | list[int],
        readable_timedelta: bool = False,
    ) -> tuple:
        """Interprets a patient timeline matrix into a series of plain texts.
        Args:
            timeline_list (torch.Tensor | list[torch.Tensor]): List of timelines to be interpreted.
                A torch.Tensor object can be accepted for inference of a single timeline.
            catalog_idx_list (list): List of catalog index sequences. Alternatively, you can pass a single sequence of
                catalog indexes for inference of a single timeline.
            readable_timedelta (bool): If true, timedelta values are made into a readable text format (%Y/%m/%d %H:%M).
        Returns:
            interpreted_texts (list): Sequence of plain texts generated from the input timeline.
            interpreted_codes (list): Sequence of code values such as diagnosis codes.
            numeric_sequence (list): Sequence of numeric values.
            timedelta_indexes (list): List of indexes of timedelta values.
        """
        # Check args
        if isinstance(timeline_list, torch.Tensor):
            timeline_list = [timeline_list]
        if catalog_idx_list and isinstance(catalog_idx_list[0], int):
            catalog_idx_list = [catalog_idx_list]
        if len(timeline_list) != len(catalog_idx_list):
            raise ValueError(
                "'timeline_list' and 'catalog_idx_list' must have the same number of elements."
            )
        lengths = []
        for t, c in zip(timeline_list, catalog_idx_list):
            if t.dim() != 3:
                raise ValueError("All tensors in 'timeline_list' must be 3D tensors.")
            if t.size(1) != len(c):
                raise ValueError(
                    f"""The length of each element in 'timeline_list' and 'catalog_idx_list' must be the same.
                    (Timeline of length {t.size(1)} and catalog ids of length {len(c)} are found.)
                    """
                )
            lengths.append(len(c))

        # Interpret
        if lengths:
            timedelta_sequence = extract_timedelta_sequence(timeline_list, flatten=True)
            numerics = [t[0, :, self.timedelta_dim] for t in timeline_list]
            numeric_sequence = torch.cat(numerics, dim=0).tolist()
            flattened_catalog_indexes = []
            for lst in catalog_idx_list:
                flattened_catalog_indexes += lst
            vals = self._interpret_timeline(
                flattened_catalog_indexes,
                numeric_sequence,
                timedelta_sequence,
                lengths,
                readable_timedelta=readable_timedelta,
            )
        else:
            vals = tuple([[], [], []])
            numeric_sequence, flattened_catalog_indexes = [], []

        (
            interpreted_texts,
            interpreted_codes,
            timedelta_indexes,
        ) = vals

        return (
            interpreted_texts,
            interpreted_codes,
            numeric_sequence,
            timedelta_indexes,
            flattened_catalog_indexes,
            lengths,
        )

    def create_table(
        self,
        timeline_list: torch.Tensor | list[torch.Tensor],
        catalog_idx_list: list[list[int]] | list[int],
        readable_timedelta: bool = False,
        patient_id_start: int = 0,
    ):
        """Creates tabular data from patient timelines.
        The produced tables mimic the eval tables.
        Args:
            timeline_list (torch.Tensor | list[torch.Tensor]): List of timelines to be interpreted.
                A torch.Tensor object can be accepted for inference of a single timeline.
            catalog_idx_list (list): List of catalog indexe sequences. Alternatively, you can pass a single sequence of
                catalog indexes for inference of a single timeline.
            readable_timedelta (bool): If true, timedelta values are made into a readable text format (%Y/%m/%d %H:%M).
            patient_id_start (int): First patient ID in the table. Patient IDs are assigned continuously
                starting from this value.
        Returns:
            df (pd.DataFrame): Created table.
        """
        # Table definitions
        table_params = {
            config.COL_PID: str,
            config.COL_TYPE: int,
            config.COL_AGE: str,
            config.COL_CODE: str,
            config.COL_TEXT: str,
            config.COL_RESULT: str,
        }
        # Interpret inputs
        (
            interpreted_texts,
            interpreted_codes,
            numeric_sequence,
            timedelta_indexes,
            flattened_catalog_indexes,
            lengths,
        ) = self.interpret_timeline(timeline_list, catalog_idx_list, readable_timedelta)
        # Valdate args
        if lengths:
            # Create series of continuous patient IDs
            timeline_starts = np.hstack([np.array([0]), np.cumsum(lengths)])
            timeline_starts = timeline_starts[:-1]
            continuous_num = np.arange(0, len(lengths)) + patient_id_start
            patient_ids = np.full(len(interpreted_texts), np.nan)
            patient_ids[timeline_starts] = continuous_num
            patient_ids = pd.Series(patient_ids).ffill()
            patient_ids = patient_ids.astype(int)

            # Initialize a dataframe
            df = pd.DataFrame(
                {
                    config.COL_CODE: interpreted_codes,
                    config.COL_NUMERIC: numeric_sequence,
                    config.COL_TEXT: interpreted_texts,
                    "catalog_indexes": flattened_catalog_indexes,
                }
            )

            # Give patient id
            df[config.COL_PID] = patient_ids
            # Separate records(demographic=0, admissions=1, diagnoses=2 prescriptions=3, labs=4, eot=5)
            df[config.COL_TYPE] = -1
            # Flag for demographic data
            dmg_type = config.RECORD_TYPE_NUMBERS[config.DMG]
            sex_indexes = self.catalog_index_lists["sex_tokens"]
            sex_mask = df["catalog_indexes"].isin(sex_indexes)
            df.loc[sex_mask, config.COL_TYPE] = dmg_type
            df.loc[timeline_starts, config.COL_TYPE] = dmg_type
            # Flag for admission-related records
            adm_type = config.RECORD_TYPE_NUMBERS[config.ADM]
            dsc_type = config.RECORD_TYPE_NUMBERS[config.DSC]
            adm_idx = self.special_token_dict["[ADM]"]
            dsc_idx = self.special_token_dict["[DSC]"]
            adm_mask = df["catalog_indexes"] == adm_idx
            dsc_mask = df["catalog_indexes"] == dsc_idx
            df.loc[adm_mask, config.COL_TYPE] = adm_type
            df.loc[dsc_mask, config.COL_TYPE] = dsc_type
            # Flag for diagnosis records
            dx_type = config.RECORD_TYPE_NUMBERS[config.DX]
            dx_code_min_index = self.min_indexes[config.DX_CODE]
            dx_code_max_index = self.max_indexes[config.DX_CODE]
            dx_code_mask = df["catalog_indexes"].between(
                dx_code_min_index, dx_code_max_index
            )
            df.loc[dx_code_mask, config.COL_TYPE] = dx_type
            # Flag for prescription record
            pharma_type = config.RECORD_TYPE_NUMBERS[config.PSC_O]
            med_code_min_index = self.min_indexes[config.MED_CODE]
            med_code_max_index = self.max_indexes[config.MED_CODE]
            med_code_mask = df["catalog_indexes"].between(
                med_code_min_index, med_code_max_index
            )
            df.loc[med_code_mask, config.COL_TYPE] = pharma_type
            # Flag for laboratory values
            lab_type = config.RECORD_TYPE_NUMBERS[config.LAB_R]
            jlac_min_index = self.min_indexes[config.LAB_CODE]
            jlac_max_index = self.max_indexes[config.LAB_CODE]
            jlac_mask = df["catalog_indexes"].between(jlac_min_index, jlac_max_index)
            df.loc[jlac_mask, config.COL_TYPE] = lab_type
            # Flag for end of timelines
            eot_type = config.RECORD_TYPE_NUMBERS[config.EOT]
            eot_mask = df["catalog_indexes"] == self.special_token_dict["[EOT]"]
            df.loc[eot_mask, config.COL_TYPE] = eot_type

            # Handle timedelta
            df[config.COL_AGE] = None
            df.loc[timedelta_indexes, config.COL_AGE] = df.loc[
                timedelta_indexes, config.COL_TEXT
            ]
            df[config.COL_AGE] = df[config.COL_AGE].ffill()
            df.loc[timedelta_indexes, config.COL_TEXT] = ""
            # Handle numerics
            numeric_mask = ~(df[config.COL_NUMERIC].isna())
            shifted_numerics = df[config.COL_TEXT].mask(~numeric_mask, None).shift(-1)
            df[config.COL_RESULT] = shifted_numerics
            df[config.COL_TEXT] = df[config.COL_TEXT].mask(numeric_mask, "")
            # Handle nonnumeric test results
            max_nonnum_idx = self.max_indexes["nonnumeric_lab_values"]
            min_nonnum_idx = self.min_indexes["nonnumeric_lab_values"]
            nonnum_mask = df["catalog_indexes"].between(min_nonnum_idx, max_nonnum_idx)
            nonnum_values = df[config.COL_TEXT].mask(~nonnum_mask, None).shift(-1)
            df[config.COL_RESULT] = df[config.COL_RESULT].fillna(nonnum_values)
            df[config.COL_TEXT] = df[config.COL_TEXT].mask(nonnum_mask, "")
            # Handle discharge records
            max_ddi = self.max_indexes["discharge_disposition_tokens"]
            min_ddi = self.min_indexes["discharge_disposition_tokens"]
            ddi_mask = df["catalog_indexes"].between(min_ddi, max_ddi)
            dispositions = df[config.COL_CODE].mask(~ddi_mask, None).shift(-1)
            df[config.COL_RESULT] = df[config.COL_RESULT].fillna(dispositions)
            df[config.COL_TEXT] = df[config.COL_TEXT].mask(ddi_mask, "")
            # Drop unnecessary rows
            df[config.COL_TEXT] = df[config.COL_TEXT].fillna("")
            df = df[df[config.COL_TEXT] != ""]
            df[config.COL_RESULT] = df[config.COL_RESULT].fillna("")
            # Drop columns
            df = df[list(table_params.keys())]

        else:
            df = pd.DataFrame(columns=list(table_params.keys()))

        # Data type checking
        df = format_df(df, table_params=table_params)

        return df
