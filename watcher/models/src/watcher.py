"""The main Watcher model"""

import logging
from datetime import timedelta
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from .layers import (
    WatcherEncoder,
    TransformerLayerStack,
    LogitsHead,
)
from .interpreter import WatcherInterpreter
from ...general_params import watcher_config as config
from ...utils import (
    extract_timedelta_sequence,
    compute_numeric_steps,
)


class Watcher(nn.Module):
    """
    A Watcher model class.
    """

    def __init__(
        self,
        categorical_dim: int,
        input_vocab_size: int,
        embedding_dim: int,
        max_sequence_length: int,
        num_layers: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout_rate: float,
        n_numeric_bins: int,
        td_small_step: int,
        td_large_step: int,
        precision: Literal["float32", "float16", "bfloat16"] = "bfloat16",
        catalogs_dir: str = None,
        lab_stats_dir: str = None,
    ):
        """Model initialization.

        Args:
            categorical_dim (int): Dimension for categorical values.
            input_vocab_size (int): Vocabulary size of the categorical embedding layer.
            embedding_dim (int): Dimension of the input embeddings.
            max_sequence_length (int): Maximum length of the input sequence.
            num_layers (int): N transformer blocks in the model.
            num_heads (int): Number of attention heads in the transformer layers.
            ff_hidden_dim (int): Hn dimension of the feed-forward layers.
            dropout_rate (float): Dropout rate for model's dropout layers.
            n_numeric_bins (int): Number of bins for numeric values.
            td_small_step (int): Step size for the timedelta within 24 hours.
            td_large_step (int): Step size for the timedelta beyond 24 hours.
            catalogs_dir (str): Absolute path to the directory where the catalog and catalog info exist.
            lab_stats_dir (str): Phe directory where the laboratory test result stats exist.
        """
        super().__init__()

        # *********************
        # * General variables *
        # *********************
        self.n_demographic_rows = config.DEMOGRAPHIC_ROWS
        self.timedelta_dim = len(config.TIMEDELTA_COMPONENT_COLS)
        self.numeric_dim = config.NUMERIC_DIM
        self.input_vocab_size = input_vocab_size
        self.categorical_dim = categorical_dim
        self.input_dim = (
            self.timedelta_dim + self.numeric_dim + self.categorical_dim + 1
        )  # <- +1 for admission status
        self.numeric_start = self.timedelta_dim
        self.categorical_start = self.numeric_start + self.numeric_dim
        self.adm_status_start = self.categorical_start + self.categorical_dim
        self.embedding_dim = embedding_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.max_sequence_length = max_sequence_length
        self.n_numeric_bins = n_numeric_bins
        self.td_small_step = td_small_step
        self.td_large_step = td_large_step
        padding_row = torch.full((self.input_dim,), torch.nan)
        padding_row[self.categorical_start :] = (
            0  # <- Admission status defaults to '0'.
        )
        self.register_buffer("padding_row", padding_row, persistent=False)
        if (catalogs_dir is not None) and (lab_stats_dir is not None):
            # Instantiate an interpreter object
            self.interpreter = WatcherInterpreter(
                n_numeric_bins=n_numeric_bins,
                catalogs_dir=catalogs_dir,
                lab_stats_dir=lab_stats_dir,
            )
            # Set frequently appearing indexes as attributes
            self.eot_index = self.interpreter.special_token_dict["[EOT]"]
            self.adm_index = self.interpreter.special_token_dict["[ADM]"]
            self.dsc_index = self.interpreter.special_token_dict["[DSC]"]
            self.lab_token_index = self.interpreter.special_token_dict["[LAB]"]
            # Vocab size
            self.effective_total_vocab_size = len(
                self.interpreter.catalog_index_lists["full"]
            )
            # Timedelta steps
            timedelta_catalog = self.interpreter.catalogs["timedelta"]
            self.n_small_steps = 60 * 24 // self.td_small_step
            self.large_step_start_idx = timedelta_catalog.index[self.n_small_steps]
            self.small_step_start_idx = timedelta_catalog.index[0]
        else:
            # Instantiate model for debug
            self.interpreter = None
            self.effective_total_vocab_size = 12000

        # Precision
        if precision.lower() == "bfloat16":
            self.precision = torch.bfloat16
        elif precision.lower() == "float16":
            self.precision = torch.float16
        elif precision.lower() == "float32":
            self.precision = torch.float32

        # Initialize values necessary only for inference
        self.inf_objects = {}
        self.inf_settings = {}
        self.inf_products = {}
        self.inference_mode = False
        self.logits_filters = None
        self.small_step_starts = None
        self.small_step_noises = None
        self.large_step_generation_ref = None
        self.large_step_noises = None
        self.small_tds, self.large_tds = None, None
        self.register_buffer("categorical_input_catalog", None, persistent=False)
        self.register_buffer("numeric_input_catalog", None, persistent=False)

        # **********
        # * Layers *
        # **********
        # Trainables
        self.encoder = WatcherEncoder(
            max_sequence_length=max_sequence_length,
            vocab_size=self.input_vocab_size,
            embedding_dim=embedding_dim,
            categorical_dim=categorical_dim,
            numeric_dim=self.numeric_dim,
            timedelta_dim=self.timedelta_dim,
            padding_idx=0,
            epsilon=1e-10,
        )
        self.transformer_stack = TransformerLayerStack(
            num_layers,
            embedding_dim,
            num_heads,
            ff_hidden_dim,
            dropout_rate,
            max_sequence_length,
        )
        self.logits_head = LogitsHead(
            vocab_size=self.effective_total_vocab_size,
            input_dim=embedding_dim,
            ff_hidden_dim=ff_hidden_dim,
        )

        # Initialize weights
        self._init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_weights(self, mean: float = 0.0, std: float = 0.02):
        """Initializes model's parameters.
        Ensure all layers are named properly.
        Args:
            mean (float): Mean value for initialization. Default is 0.0.
            std (float): Standard deviation for initialization. Default is 0.02
        """
        # Count number of residual layers
        n_resid = 0
        for name, p in self.named_parameters():
            if "resid" in name and "weight" in name:
                n_resid += 1
        resid_std = std * (n_resid**-0.5)
        for name, p in self.named_parameters():
            # Params with dimensions >=2
            if p.dim() > 1:
                # Initialize layernorm weights with ones
                if "norm" in name and "weight" in name:
                    nn.init.constant_(p, 1.0)
                # Scale wights for residual layers
                elif "resid" in name and "weight" in name:
                    nn.init.normal_(p, mean=mean, std=resid_std)
                # Set weights of the padding embedding to zeros
                elif "embedding" in name:
                    nn.init.normal_(p, mean=mean, std=std)
                    p.data[0].zero_()
                # Initialize params that are summed
                elif "position_encoding" in name or "admission_encoding" in name:
                    nn.init.normal_(p, mean=mean, std=std / 2)
                # Initialize all other params
                else:
                    nn.init.normal_(p, mean=mean, std=std)
            # Params with only one dim (i.e, biases, scales)
            else:
                if name.endswith(".bias"):
                    nn.init.constant_(p, 0.0)

    def _create_logits_filters(self) -> dict:
        """Creates filters to selectively compute logits.
        Returns:
            dict: Dictionary containing various logits filters for different token categories.
        """
        # Create filters for inference
        full_index_set = set(self.interpreter.catalog_index_lists["full"])
        logits_filters = {}
        logits_filters["default"] = [0]
        logits_filters["sex_tokens"] = full_index_set - set(
            self.interpreter.catalog_index_lists["sex_tokens"]
        )
        logits_filters["diagnoses"] = full_index_set - set(
            self.interpreter.catalog_index_lists[config.DX_CODE]
        )
        logits_filters["drugs"] = full_index_set - set(
            self.interpreter.catalog_index_lists[config.MED_CODE]
        )
        logits_filters["labs"] = full_index_set - set(
            self.interpreter.catalog_index_lists[config.LAB_CODE]
        )
        logits_filters["numeric"] = full_index_set - set(
            self.interpreter.catalog_index_lists["numeric_lab_values"]
        )
        logits_filters["nonnumerics"] = full_index_set - set(
            self.interpreter.catalog_index_lists["nonnumeric_lab_values"]
        )
        logits_filters["lab_results"] = (
            full_index_set
            - set(self.interpreter.catalog_index_lists["nonnumeric_lab_values"])
            - set(self.interpreter.catalog_index_lists["numeric_lab_values"])
        )
        logits_filters["ignore_lab_results"] = (
            [0]
            + self.interpreter.catalog_index_lists["nonnumeric_lab_values"]
            + self.interpreter.catalog_index_lists["numeric_lab_values"]
        )
        logits_filters["labs_and_results"] = (
            full_index_set
            - set(self.interpreter.catalog_index_lists[config.LAB_CODE])
            - set(self.interpreter.catalog_index_lists["nonnumeric_lab_values"])
            - set(self.interpreter.catalog_index_lists["numeric_lab_values"])
        )
        logits_filters["timedelta"] = full_index_set - set(
            self.interpreter.catalog_index_lists["timedelta"]
        )
        logits_filters["timedelta_within_24hr"] = full_index_set - set(
            range(self.small_step_start_idx, self.large_step_start_idx)
        )
        logits_filters["ignore_timedelta"] = [0] + self.interpreter.catalog_index_lists[
            "timedelta"
        ]
        logits_filters["endless"] = [
            0,
            self.eot_index,
            self.interpreter.catalog_index_lists["timedelta"][-1],
        ]
        for k, v in logits_filters.items():
            if isinstance(v, set):
                v = list(v)
            logits_filters[k] = v
        return logits_filters

    def _create_categorical_input_vectors(self) -> Tensor:
        """Creates catalogs for input vectors for categorical values.
        Admission status (categorical_input_catalog[:,-1]) defaults to 0.
        """
        categorical_cols = [f"c{i}" for i in range(self.categorical_dim)]
        categorical_indexes = self.interpreter.catalogs["all_categorical"]
        categorical_indexes = categorical_indexes[categorical_cols]
        categorical_indexes = torch.tensor(categorical_indexes.to_numpy())
        categorical_input_catalog = self.padding_row.repeat(
            categorical_indexes.size(0), 1
        )
        categorical_input_catalog[:, self.categorical_start : -1] = categorical_indexes
        return categorical_input_catalog

    def _create_numeric_input_vectors(self) -> Tensor:
        """Creates catalogs for input vectors for numeric values."""
        # Percentile strategy
        num_steps = compute_numeric_steps(num_bins=self.n_numeric_bins)
        num_steps = torch.from_numpy(num_steps).to(self.device)
        numeric_input_catalog = self.padding_row.repeat(num_steps.size(0), 1)
        numeric_input_catalog[:, self.numeric_start] = num_steps

        return numeric_input_catalog

    def _create_timedelta_vocab_series(self) -> tuple:
        """Prepare series of timedelta vocabularies.
        These products are used in '_create_time_anchor_mask' to select valid vocabularies for timedelta prediction.
        Note that the timedelta values in the series are the 'MAXIMUM' value in the timedelta bins.
        """
        # NOTE: Last timedelta vocab (out-of-range timedelta) is not included in the series.

        def _large_td_series_from_text(td_catalog: pd.DataFrame) -> pd.Series:
            td_df = (
                td_catalog[config.COL_TEXT]
                .str.extract(r"(\d+) days \& \d{2}:\d{2}-(\d{2}):(\d{2})")
                .dropna()
            )
            td_df.columns = ["d", "h", "m"]
            td_df = td_df.dropna(how="any")
            td_df = td_df.astype(int)
            tds = td_df["d"] * 24 * 60 + td_df["h"] * 60 + td_df["m"]
            tds = pd.to_timedelta(tds, unit="min")
            # Convert to np.timedelta64[m] series
            tds = tds.to_numpy().astype("timedelta64[m]")
            return tds

        def _small_td_series_from_text(td_catalog: pd.DataFrame) -> pd.Series:
            td_df = td_catalog[config.COL_TEXT].str.extract(r"(\d+) minutes").dropna()
            td_df.columns = ["m"]
            td_df = td_df.dropna(how="any")
            td_df = td_df.astype(int)
            tds = pd.to_timedelta(td_df["m"], unit="min")
            # Convert to np.timedelta64[m] series
            tds = tds.to_numpy().astype("timedelta64[m]")
            return tds

        # Get timedelta catalog
        td_catalog = self.interpreter.catalogs["timedelta"]
        # Timedelta series for small steps
        small_tds = _small_td_series_from_text(td_catalog)
        # Timedelta series for large steps
        large_tds = _large_td_series_from_text(td_catalog)

        return small_tds, large_tds

    def _pad_to_max_seq_len(self, x: Tensor) -> Tensor:
        """Pads input to the max sequence length"""
        if x.size(1) < self.max_sequence_length:
            out = torch.cat(
                [
                    x,
                    self.padding_row.repeat(
                        x.size(0), self.max_sequence_length - x.size(1), 1
                    ),
                ],
                dim=1,
            )
            return out
        return x

    # *******************
    # * Forward methods *
    # *******************
    def forward(
        self,
        x: Tensor,
        use_kv_cache: bool = False,
        record_attention_weights: bool = False,
    ) -> Tensor:
        """Forward pass.
        Args:
            x (Tensor): Input.
            use_kv_cache (bool): If true, KV-cache is used.
                This method automatically clears cache if the input length grows longer than the max sequence length.
            record_attention_weights (bool): If true, attention weights are stored in models in the transformer
                layer stack.
        Returns:
            out (Tensor): Shape [N, L, vocab size]
        """
        # Encode inputs
        # This step is done in the original precision
        out = self.encoder(x)

        # Transformer stack & prediction head
        with torch.autocast(device_type="cuda", dtype=self.precision, enabled=True):
            # Transformer layers
            out = self.transformer_stack(out, use_kv_cache, record_attention_weights)
            if self.inference_mode:
                out = out[:, -1:]
            # Compute logits
            out = self.logits_head(out)

        return out

    # *************************
    # * Methods for inference *
    # *************************
    # TODO (Yu Akagi): consider taking care of catalog indexes here as well
    def remove_padding_rows(self, t: Tensor, c: list) -> Tensor:
        """Detects and removes padding rows from the given matrix.
        Args:
            t (Tensor): This is a single patient timeline.
                The size is expected to be (1, sequence length, input dim).
            c (list): Catalog indexes. A list of integers.
        Returns:
            cleaned_t (Tensor): 3D tensor without paddings.
            cleaned_c (list): Catalog indexes without paddings.
        """
        # Create boolean mask for paddings
        with torch.device(device=t.device):
            td_is_pad = torch.isnan(t[0, :, 0])
            num_is_pad = torch.isnan(t[0, :, self.timedelta_dim])
            categ_is_pad = t[0, :, self.categorical_start].long() == 0
            is_pad = td_is_pad & num_is_pad & categ_is_pad
            if is_pad.any():
                non_pad = ~is_pad
                # Remove paddings from timeline
                cleaned_t = t[:, non_pad, :]
                # Remove paddings from catalog indexes
                cleaned_c = torch.tensor(c)[non_pad[: len(c)]].tolist()
                return cleaned_t, cleaned_c
            else:
                return t, c

    def eval(self, for_inference: bool = False):
        """Switches the model to evaluation mode with optional inference preparation.

        This method extends PyTorch's `eval()` function. If `for_inference` is set to `True`,
        it additionally initializes settings and buffers required for inference.

        Args:
            for_inference (bool, optional):
                - If `False`, defaults to PyTorch's original `self.eval()` behavior.
                - If `True`, prepares the model for inference by setting up various
                  filters, catalogs, and timedelta generation templates.
                  Default is `False`.

        Behavior:
            - When `for_inference` is `True`:
              - Creates and registers necessary filters and buffers.
              - Initializes timedelta generation templates for both small and large steps.
              - Prepares timedelta vocabulary series for prediction.
              - Sets an `inference_mode` flag for inference readiness.
        """
        # Original eval()
        super().eval()

        if for_inference:
            self.logits_filters = self._create_logits_filters()
            self.register_buffer(
                "categorical_input_catalog",
                self._create_categorical_input_vectors(),
                persistent=False,
            )
            self.register_buffer(
                "numeric_input_catalog",
                self._create_numeric_input_vectors(),
                persistent=False,
            )

            # Create templates for timedelta value generation (Small steps)
            timedelta_catalog = self.interpreter.catalogs["timedelta"]
            self.small_step_starts = (
                np.arange(self.n_small_steps) * self.td_small_step
            ).astype("timedelta64[m]")
            self.small_step_noises = np.arange(self.td_small_step).astype(
                "timedelta64[m]"
            )
            # Create templates for timedelta value generation (Large steps)
            # TODO: Consider using numpy only for this step without pandas.
            large_step_catalog = timedelta_catalog.loc[
                self.large_step_start_idx : self.interpreter.max_indexes["timedelta"]
                - 1
            ].copy()
            large_td_parts = large_step_catalog[config.COL_TEXT].str.extract(
                r"\+ (\d+)(?:-(\d+))? days \& (\d{2}):(\d{2})-(\d{2}):(\d{2})"
            )
            large_td_parts.columns = [
                "start_days",
                "end_days",  # <- may contain nan
                "start_hours",
                "start_minutes",
                "end_hours",
                "end_minutes",
            ]
            large_td_parts["end_days"] = large_td_parts["end_days"].fillna(
                large_td_parts["start_days"]
            )
            large_td_parts = large_td_parts.astype(int)
            large_td_parts["starting_time"] = (
                large_td_parts["start_minutes"] + large_td_parts["start_hours"] * 60
            )
            large_td_parts["starting_time"] = pd.to_timedelta(
                large_td_parts["starting_time"], unit="m"
            )
            self.large_step_generation_ref = large_td_parts[
                ["start_days", "end_days", "starting_time"]
            ]
            self.large_step_noises = np.arange(self.td_large_step).astype(
                "timedelta64[m]"
            )

            # Timedelta vocabulary series for timedelta prediction
            self.small_tds, self.large_tds = self._create_timedelta_vocab_series()

            # Set a flag for inference initialization
            self.inference_mode = True

    def setup_cache(self, batch_size: int):
        """Initializes KV-cache."""
        with torch.device(device=self.device):
            self.transformer_stack.setup_cache(batch_size, dtype=self.precision)

    def empty_cache(self):
        """Empties KV-cache with zeros."""
        self.transformer_stack.empty_cache()

    def trim_cache(self, actives: Tensor, new_size: int):
        """Trims cache."""
        self.transformer_stack.trim_cache(actives, new_size)

    def delete_cache(self):
        """Deletes KV-cache."""
        self.transformer_stack.delete_cache()

    @property
    def cache_length(self):
        """Gets the current KV-cache length"""
        return self.transformer_stack.cache_length

    def get_attention_map(self) -> Tensor:
        """Loads attention map from transformer stack.

        The returned map is the simple average of maps from all the transformer blocks.
        """
        attentions = torch.cat(
            [
                layer.attention.attention_weights
                for layer in self.transformer_stack.layers
            ],
            dim=0,
        )
        attention_map = torch.mean(attentions, dim=0)
        return attention_map

    def truncate_timeline(
        self, x: Tensor, catalog_ids: list | None = None, length: int | None = None
    ) -> Tensor | tuple[Tensor, list]:
        """Truncate long timelines for inference.
        Args:
            x (Tensor): Input tensor,
            catalog_ids (list[int]): Catalog indexes that is paired with the input.
                If this is passed, then it is also truncated so that it matches the truncated input tensor.
            length (int): Length to which the input is truncated.
        Returns:
            Tensor or tuple[Tensor, list]: If catalog indexes are passed, returns both truncated tensor and catalog indexes;
                otherwise, only the truncated tensor.
        """
        if length is None:
            length = self.max_sequence_length
        if x.size(1) > length:
            demographic_rows = x[:, : self.n_demographic_rows, :]
            slice_point = self.n_demographic_rows - length
            latest_rows = x[:, slice_point:, :]
            truncated_x = torch.cat([demographic_rows, latest_rows], dim=1)
            if catalog_ids is not None:
                truncated_indexes = (
                    catalog_ids[: self.n_demographic_rows] + catalog_ids[slice_point:]
                )
                return truncated_x, truncated_indexes

            return truncated_x
        # If the timeline is shorter than the length, return the inputs without truncation
        else:
            if catalog_ids is not None:
                return x, catalog_ids
            return x

    def create_timedelta_rows(self, timedelta_series: np.ndarray) -> Tensor:
        """Creates timedelta input tensors from timedelta series.

        Args:
            timedelta_series (np.ndarray): Array of np.timedelta64.
        Returns:
            td_rows (Tensor): Timeldeta input tensors.
        """
        # Validate dtype
        if timedelta_series.dtype != "timedelta64[m]":
            tds = timedelta_series.dtype("timedelta64[m]")
        else:
            tds = timedelta_series
        # Separate days and minutes
        td_days = tds.astype("timedelta64[D]")
        td_mins = tds - td_days
        # Create tensor
        td_series = pd.Series(timedelta_series)
        days = torch.from_numpy(td_days.astype(int))
        mins = torch.from_numpy(td_mins.astype(int))
        # Finalize the input row
        td_rows = self.padding_row.repeat(len(td_series), 1)
        td_rows[:, 0] = days // 365 / 120
        td_rows[:, 1] = days % 365 // 30 / 12
        td_rows[:, 2] = days % 365 % 30 / 29
        td_rows[:, 3] = mins // 60 / 23
        td_rows[:, 4] = mins % 60 / 59
        return td_rows

    def preprocess_prompt(
        self,
        timeline: Tensor,
        catalog_ids: list[int],
        time_anchor: timedelta | None = None,
    ) -> tuple[Tensor, list, timedelta]:
        """Cleans a prompt before being used for autoregressive inference.

        A prompt is a pair of timeline and series of catalog ids.

        This function involves the following steps:
            - 1. Move the prompt timeline to model's device
            - 1. Remove paddings from the prompt
            - 2. Get the latest time (patient age) from the timeline
            - 3. Validates the prompt for time anchoring, and modifies the prompt if indicated.

        Args:
            timeline (Tensor): Prompt timeline with the batch size of 1.
            catalog_ids (list[int]): Prompt catalog ids that is paired with `timeline`.
            time_anchor (timedelta|None): Oldest possible time that the generated timeline can start from.
                If none, this function skips validation of the prompt for time anchoring.
        Returns:
            cleaned_t (Tensor): Cleaned timeline. If the input `timeline` is already clean, `cleaned_t` is the same with `timeline`.
            cleaned_c (list[int]): Cleaned catalog ids. If the input `catalog_ids` is already clean, `cleaned_c` is the same with `catalog_ids`.
            latest_time (timedelta): Latest time (patient age) in the timeline.
        """
        # Check device
        cleaned_t = timeline.to(self.device)
        # Remove paddings from the prompt
        cleaned_t, cleaned_c = self.remove_padding_rows(cleaned_t, catalog_ids)
        # Preprocess
        with torch.device(self.device):
            # Get the latest timedelta of the given patient timeline
            timedelta_sequences = extract_timedelta_sequence(cleaned_t)
            latest_time = timedelta_sequences[0][-1]
            # *******************************************
            # * Extra handlings for timedelta anchoring *
            # *******************************************
            # NOTE: Inputs are properly handled here for timedelta anchoring in order to avoid errors during inference
            if time_anchor is not None:
                # Mask for td anchor
                td_anchored = time_anchor > latest_time
                if td_anchored:
                    # Exception No1: If the last row is timedelta, then remove it.
                    # NOTE: This step must come before handling of too-short inputs, because this step may yield too-short inputs.
                    # NOTE: Do not include timeliens shorter than or equal to the number of demographic rows, because it may result in timelines with zero lengths.
                    ends_with_td = (not cleaned_t[0, -1, 0].isnan().item()) and (
                        cleaned_t.size(1) > self.n_demographic_rows
                    )
                    if ends_with_td:
                        logging.warning(
                            """
                            Warning! The input timeline ends with a timedelta (age) row but time_anchor is set.
                            The last row is removed from the timeline.
                            """
                        )
                        # 1. Remove the last row from the input
                        cleaned_t = cleaned_t[:, :-1]
                        # 2. Remove the last catalog id
                        cleaned_c = cleaned_c[:-1]
                        # 4. Update the timedelta
                        new_tds = extract_timedelta_sequence(cleaned_t)
                        latest_time = new_tds[0][-1]

                    # Exception No2: If the input is shorter than n_demographic_rows, then update the initial age to the anchor
                    too_short = cleaned_t.size(1) <= self.n_demographic_rows
                    if too_short:
                        logging.warning(
                            """
                            Warning! The input timeline only contains demographics but time_anchor is set.
                            The initial age of the patient is modified accordingly.
                            """
                        )
                        first_age_days = timedelta(days=time_anchor.days)
                        latest_time = first_age_days
                        new_init_row = self.create_timedelta_rows(
                            np.array([latest_time], dtype=object).astype(
                                "timedelta64[m]"
                            ),
                        )
                        cleaned_t[:, 0, :] = new_init_row
        return cleaned_t, cleaned_c, latest_time

    def _create_time_anchor_mask(
        self, current_time: np.ndarray, target_time: timedelta
    ) -> Tensor:
        """Creates a mask for batched logits to force the next predictions to be timedelta values
        beyond the target time.
        Args:
            current_time (np.ndarray): Series of current timedelta.
            target_time (timedelta): Target time.
        Returns:
            mask (Tensor): Boolean mask with the size of (batch_size, effective_total_vocab_size).
                This mask can be used to mask out logits.
                A true value in this maks indicates masking out.
        """
        with torch.device(self.device):
            # Pick up rows
            td_anchored = current_time < target_time
            td_anchored_idxs = td_anchored.nonzero()[0].tolist()
            n_anchored = len(td_anchored_idxs)
            # Check for small timedelta steps
            diff_in_minutes = target_time - current_time[td_anchored]
            small_td_mtx = self.small_tds.reshape(1, -1).repeat(n_anchored, 0)
            small_td_mask = small_td_mtx < diff_in_minutes.reshape(-1, 1)
            # Check for large timedelta steps
            current_td_days = current_time.astype("timedelta64[D]")
            diff_in_days = target_time - current_td_days[td_anchored]
            large_td_mtx = self.large_tds.reshape(1, -1).repeat(n_anchored, 0)
            large_td_mask = large_td_mtx < diff_in_days.reshape(-1, 1)
            # Concatenate the mask
            td_valid_mask = torch.tensor(np.hstack([small_td_mask, large_td_mask]))
            # Replace the main mask
            batch_size = current_time.shape[0]
            mask = torch.full((batch_size, self.effective_total_vocab_size), False)
            mask[td_anchored_idxs, self.small_step_start_idx : -1] = td_valid_mask
            mask[td_anchored_idxs, : self.small_step_start_idx] = True
            mask[td_anchored_idxs, -1] = False
        return mask

    def compute_probs(
        self,
        input_tensor: Tensor,
        last_ids: Tensor,
        pos: Tensor,
        logits: Tensor,
        current_time: np.ndarray | None = None,
        time_anchor: timedelta | None = None,
        logits_filter: str = "default",
        temperature: float = 1.0,
    ) -> Tensor:
        """Computes confidence scores.
        Args:
            input_tensor (Tensor): Input timeline tensor.
            last_ids (Tensor): Batch of ids at the previous position.
            pos (Tensor): Current position in the input timeline tensor.
                This must be from 0 to self.max_sequence_length.
            logits (Tensor): Logits computed by `self.forward()`.
                The shape is either [N, vocab size] or [N, 1, vocab size].
            current_time (np.ndarray): Current timedelta.
                This must not be None if `time_anchor` is specified.
            time_anchor (timedelta|None): Oldest possible time that the generated timeline can start from.
            logits_filter (str): Filter to select logits.
            temperature (float): Temperature to scale logits.
        Returns:
            probs (Tensor): Shape [N, vocab size].
                A batch of confidence scores over model's vocabulary.
        """
        # Set up
        if pos == -1:
            pos = torch.tensor(input_tensor.size(1) - 1)
        adm_status = input_tensor[:, pos, -1]
        is_inpatient = adm_status.long().bool()
        is_outpatient = ~is_inpatient

        # Check input shape
        if logits.dim() == 3:
            logits = logits.view(logits.size(0), -1)
        # Set the default device
        with torch.device(self.device):
            strict = True
            if strict:
                # *******************
                # * Syntax checking *
                # *******************
                # Apply minimum filter
                filtered_indexes = self.logits_filters[logits_filter]
                logits[:, filtered_indexes] = float("-inf")

                # Apply additional filters for syntax compliance
                # Rule No.0: Patient sex generation
                generate_patient_sex = pos == 0
                if generate_patient_sex:
                    filtered_indexes = self.logits_filters["sex_tokens"]
                    logits[:, filtered_indexes] = float("-inf")
                # TODO (Yu Akagi): Enhance this process (discharge dispositions, prohhibit successive timedelta, etc...)
                # Rule No.1: First actual timedelta change must be within 24 hours
                is_first_td = pos == (config.DEMOGRAPHIC_ROWS - 1)
                if is_first_td:
                    filtered_indexes = self.logits_filters["timedelta_within_24hr"]
                    logits[:, filtered_indexes] = float("-inf")
                # Rule No.2: Only lab test results can follow lab test codes
                max_lab_idx = self.interpreter.max_indexes[config.LAB_CODE]
                min_lab_idx = self.interpreter.min_indexes[config.LAB_CODE]
                lab_mask = (last_ids >= min_lab_idx) & (last_ids <= max_lab_idx)
                if lab_mask.any():
                    filtered_indexes = self.logits_filters["lab_results"]
                    logits[:, filtered_indexes] = logits[
                        :, filtered_indexes
                    ].masked_fill(lab_mask.view(-1, 1), float("-inf"))
                # Rule No.3: Lab test results should not appear without a preceding lab codes
                non_lab_mask = ~lab_mask
                if non_lab_mask.any():
                    filtered_indexes = self.logits_filters["ignore_lab_results"]
                    logits[:, filtered_indexes] = logits[
                        :, filtered_indexes
                    ].masked_fill(non_lab_mask.view(-1, 1), float("-inf"))
                # Rule No.4: Timedelta rows do not appear successively
                max_td_idx = self.interpreter.max_indexes["timedelta"]
                min_td_idx = self.interpreter.min_indexes["timedelta"]
                td_mask = (last_ids >= min_td_idx) & (last_ids <= max_td_idx)
                if td_mask.any():
                    filtered_indexes = self.logits_filters["ignore_timedelta"]
                    logits[:, filtered_indexes] = logits[
                        :, filtered_indexes
                    ].masked_fill(td_mask.view(-1, 1), float("-inf"))
                # Rule No.5: Admissions and discharges alternate, EOT does not appear during admission
                if is_inpatient.any():
                    filtered_indexes = [self.adm_index, self.eot_index]
                    logits[:, filtered_indexes] = logits[
                        :, filtered_indexes
                    ].masked_fill(is_inpatient.view(-1, 1), float("-inf"))
                if is_outpatient.any():
                    filtered_indexes = [self.dsc_index]
                    logits[:, filtered_indexes] = logits[
                        :, filtered_indexes
                    ].masked_fill(is_outpatient.view(-1, 1), float("-inf"))

            # ***********************
            # * Timedelta anchoring *
            # ***********************
            # NOTE: 'preprocess_prompt()' validates and cleans inputs before timdelta anchoring.
            if time_anchor is not None:
                td_anchored = time_anchor > current_time
                if td_anchored.any():
                    # Ensure that timedelta values can follow (prioritizing syntax compliance over forcing timedelta prediction)
                    # TODO (Yu Akagi): Make this step to handle some other situations (input ending with discharge without dispositons, etc.)
                    if pos < self.n_demographic_rows:
                        pass
                    else:
                        td_anchor_mask = self._create_time_anchor_mask(
                            current_time=current_time,
                            target_time=time_anchor,
                        )
                        anchor_skipped = lab_mask
                        td_anchor_mask[anchor_skipped, :] = False
                        logits = logits.masked_fill(td_anchor_mask, float("-inf"))
            # *****************************
            # * Compute confidence scores *
            # *****************************
            if temperature == 0:
                one_hot_index = torch.argmax(logits, dim=1)
                batch_numbers = torch.arange(0, logits.size(0))
                probs = torch.zeros_like(logits)
                probs[batch_numbers, one_hot_index] = 1.0
            else:
                if temperature != 1:
                    logits = logits / temperature
                probs = nn.functional.softmax(logits, dim=1)

        return probs

    def sample_from_probs(self, probs: Tensor) -> Tensor:
        """Samples ids from the computed probability over model's vocabulary,

        Args:
            probs (Tensor): Confidence scores computed by `compute_probs()`.
        Returns:
            sampled_ids (Tensor): Series of predicted next catalog ids. This is a 1-D array.
        """
        # Smaple ids from the probability distribution
        sampled_ids = torch.multinomial(probs, num_samples=1, replacement=False)
        # Reshape
        sampled_ids = sampled_ids.squeeze(-1).long()
        # Replace predictions of out-of-range timedelta with [EOT]
        sampled_ids[sampled_ids == self.interpreter.max_indexes["timedelta"]] = (
            self.eot_index
        )

        return sampled_ids

    def generate_next_input(
        self,
        input_tensor: Tensor,
        pos: Tensor,
        current_time: np.ndarray,
        sampled_ids: torch.Tensor,
        time_anchor: timedelta = None,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Creates rows for the next autoregressive inference with batched inputs.
        Args:
            input_tensor (Tensor): Input timeline tensor.
            pos (Tensor): Current position in the input timeline tensor.
                This must be from 0 to self.max_sequence_length.
            sampled_ids (torch.Tensor): Series of catalog ids sampled by `sample_from_probs()`.
            current_time (np.ndarray): Series of the latest timedeltas.
            time_anchor (timedelta|None): Oldest possible time that the generated timeline can start from.
        Returns:
            next_inputs (torch.Tensor): Rows that can be used for the next autoregressive inference.
            current_time (np.ndarray): Updated current time.
        """
        if pos == -1:
            pos = torch.tensor(input_tensor.size(1) - 1)
        # Segragete predictions by prediction types (masks are torch.Tensor)
        categorical_mask = (
            sampled_ids <= self.interpreter.max_indexes["all_categorical"]
        )
        numeric_mask = (
            sampled_ids <= self.interpreter.max_indexes["numeric_lab_values"]
        ) & (sampled_ids > self.interpreter.max_indexes["all_categorical"])
        td_small_mask = (sampled_ids < self.large_step_start_idx) & (
            sampled_ids > self.interpreter.max_indexes["numeric_lab_values"]
        )
        td_large_mask = (sampled_ids >= self.large_step_start_idx) & (
            sampled_ids < self.interpreter.max_indexes["timedelta"]
        )
        # Create lists of indexes
        # NOTE: This operation involves GPU-CPU data transfer, which may slow the process
        categoricals = categorical_mask.nonzero().squeeze(-1).tolist()
        numerics = numeric_mask.nonzero().squeeze(-1).tolist()
        small_tds = td_small_mask.nonzero().squeeze(-1).tolist()
        large_tds = td_large_mask.nonzero().squeeze(-1).tolist()

        # Case 1. Categorical value predictions
        if categoricals:
            categorical_indexes = sampled_ids[categoricals]
            categorical_next_inputs = self.categorical_input_catalog[
                categorical_indexes
            ]
        else:
            categorical_next_inputs = None

        # Case 2. Numeric value predictions
        if numerics:
            numeric_indexes = sampled_ids[numerics]
            numeric_steps = (
                numeric_indexes - self.interpreter.min_indexes["numeric_lab_values"]
            )
            numeric_next_inputs = self.numeric_input_catalog[numeric_steps]
        else:
            numeric_next_inputs = None

        # Case 3. Timedelta predictions within 24hrs
        if small_tds:
            small_td_indexes = sampled_ids[small_tds]
            small_td_steps = small_td_indexes - self.small_step_start_idx
            small_td_steps = small_td_steps.tolist()
            minutes = self.small_step_starts[small_td_steps]
            noise = np.random.choice(
                self.small_step_noises, len(small_tds), replace=True
            )
            time_progress = minutes + noise
            # Ensure time is progressed
            # NOTE: Timeprogress==0 is not allowed
            time_progress = np.maximum(time_progress, np.timedelta64(1, "m"))
            new_small_tds = current_time[small_tds] + minutes + noise
            if time_anchor is not None:
                new_small_tds = np.maximum(new_small_tds, time_anchor)

            current_time[small_tds] = new_small_tds
        else:
            small_td_indexes = None
        # Case 4. Timedelta predictions over 24hrs
        if large_tds:
            large_td_indexes = sampled_ids[large_tds].tolist()
            reference_rows = self.large_step_generation_ref.loc[
                large_td_indexes
            ].reset_index(drop=True)
            days = reference_rows.apply(
                lambda x: np.random.randint(x["start_days"], x["end_days"] + 1), axis=1
            )
            days = days.values.astype("timedelta64[D]")
            minutes = reference_rows["starting_time"].values.astype("timedelta64[m]")
            noise = np.random.choice(
                self.large_step_noises, len(large_tds), replace=True
            )
            current_days = current_time[large_tds].astype("timedelta64[D]")
            new_large_tds = current_days + days + minutes + noise
            if time_anchor is not None:
                new_large_tds = np.maximum(new_large_tds, time_anchor)
            current_time[large_tds] = new_large_tds
        # Compute text timedelta input rows
        tds = small_tds + large_tds
        if tds:
            td_next_inputs = self.create_timedelta_rows(current_time[tds])
        else:
            td_next_inputs = None

        # Concatenate
        tensors = []
        positions = torch.tensor([])
        input_candidates = [
            categorical_next_inputs,
            numeric_next_inputs,
            td_next_inputs,
        ]
        position_candidates = [categoricals, numerics, tds]
        for t, p in zip(input_candidates, position_candidates):
            if t is not None:
                tensors.append(t)
                positions = torch.hstack([positions, torch.tensor(p)])
        next_inputs = torch.cat(tensors, dim=0)

        # Reorder
        reordered_pisitions = torch.argsort(positions)
        next_inputs = next_inputs[reordered_pisitions, :]

        # Update admission status
        adm_status = input_tensor[:, pos, -1]
        new_admissions = sampled_ids == self.adm_index
        new_discharges = sampled_ids == self.dsc_index
        next_inputs[:, -1] = adm_status
        next_inputs[new_admissions, -1] = 1.0
        next_inputs[new_discharges, -1] = 0.0

        return next_inputs, current_time
