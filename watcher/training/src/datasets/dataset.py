"""Dataloaders"""

import os
import math
import pickle
import random
from datetime import timedelta
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ....general_params import watcher_config as config
from ....general_params import get_settings
from ....utils import preprocess_timedelta_series, create_labels_from_timedelta_pairs


class TimelineDataset(Dataset):
    """Dataset class for Watcher model training."""

    def __init__(
        self,
        data_dir: str,
        model: torch.nn.Module,
        train: bool,
    ) -> None:
        # Initialize
        super(TimelineDataset).__init__()
        self.max_sequence_length = model.max_sequence_length
        self.n_demographic_rows = model.n_demographic_rows
        self.train = train
        self.num_workers = 1
        self.worker_id = 0
        # Load metadata
        metadata_path = os.path.join(data_dir, config.TRAJECTORY_METADATA)
        metadata = pd.read_pickle(metadata_path)
        bundle_dir = os.path.join(data_dir, config.DIR_TRAJECTORY_BUNDLES)
        if not bundle_dir.endswith("/"):
            bundle_dir += "/"
        metadata["abs_file_path"] = bundle_dir + metadata["file_path"]

        # ***** Initialization for resampling *****
        data_lengths = metadata["effective_length"].values.astype(float)
        n_sampled = data_lengths / self.max_sequence_length
        # Set limits (1<= weight <= MAX_RESAMPLING)
        n_sampled = np.clip(n_sampled, 1, config.MAX_RESAMPLING)
        # Compute weights (Do not use 'self.total' as the denominator here.)
        sampling_weights = n_sampled / n_sampled.sum()
        # Total sampling number
        self.total = int(np.ceil(n_sampled.sum()))
        # Set
        self.sampling_weights = sampling_weights
        self.all_files = metadata["abs_file_path"].values
        # Resample files for the fist time
        if get_settings("DEBUG_MODE"):
            debug_chunks = get_settings("DEBUG_CHUNKS")
            self.total = min(self.total, debug_chunks)
        self.resampled_files = []
        self.resample_data(epoch=0)
        # **************************

        # Create a template for padding.
        label_pad = torch.tensor([config.LOGITS_IGNORE_INDEX])
        padding_row = model.padding_row.clone()
        if padding_row.device != label_pad.device:
            padding_row = padding_row.to(label_pad.device)
        padding_row = torch.cat([padding_row, label_pad], dim=0)
        self.padding_template = padding_row.repeat(self.max_sequence_length, 1)
        # Collect data needed for data augmentation
        self.first_timedelta_label = model.interpreter.min_indexes["timedelta"]
        self.td_small_step = model.td_small_step
        self.td_large_step = model.td_large_step
        self.timedelta_dim = model.timedelta_dim

    def resample_data(self, epoch: int):
        """Resample dataset per epoch."""
        np.random.seed(epoch)
        self.resampled_files = np.random.choice(
            self.all_files, size=self.total, replace=True, p=self.sampling_weights
        )

    def _shuffle_and_slice(
        self,
        timeline_and_labels: torch.Tensor,
        slice_points: list[int],
        shuffle_index_candidates: list[list[int]],
    ) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Shuffles and slices a matrix before yielding.

        This function performs the following steps:
            - shuffling rows for data augmentation while preserving chronological order
            - padding matrices too short, or truncating matrices too longs
        Args:
            timeline_and_labels (torch.Tensor): Main input data for training.
            slice_points (list[int]): List of indexes where slicing can be started.
            shuffle_index_candidates (list[list[int]]): List of shuffled indexes.
                One of them is randomly selected for shuffling.
        Returns:
            processed_matrix (torch.Tensor): Matrix that is a direct input for Watcher.
            start (int): Starting index of the slicing.
            end (int): Ending index of the slicing.
                Note that 'end' is NOT included in the sliced matrix.
            n_sampled (torch.Tensor): Number of resampling.
        """
        timeline_length = timeline_and_labels.size(0)

        # Shuffle rows
        if config.SHUFFLE_INPUT:
            shuffled_indexes = random.choice(shuffle_index_candidates)
            not_yet_sliced = timeline_and_labels[shuffled_indexes, :]
        else:
            not_yet_sliced = timeline_and_labels

        # Case 1. timeline length exactly matches the output length
        if timeline_length == self.max_sequence_length:
            processed_matrix = not_yet_sliced
            start = self.n_demographic_rows
            end = self.max_sequence_length
            n_sampled = 1

        # Case 2. long timelines
        elif timeline_length > self.max_sequence_length:
            # Extract first rows as the demographic record to ensure that demographic records always come first.
            demographic = not_yet_sliced[0 : self.n_demographic_rows, :]
            # Compute number of resampling
            effective_length = len(slice_points) + self.max_sequence_length - 1
            n_sampled = effective_length / self.max_sequence_length
            n_sampled = min(n_sampled, config.MAX_RESAMPLING)

            # Select slice start and end
            # NOTE: random.randint() includes both ends
            min_start, max_start = slice_points[0], slice_points[-1]
            n_windows = math.ceil(effective_length / self.max_sequence_length)
            random_no = random.randint(0, n_windows)
            if random_no == 0:
                start = min_start
            elif random_no == 1:
                start = max_start
            else:
                start = random.choice(slice_points)
            end = start + self.max_sequence_length - self.n_demographic_rows
            # Slice
            processed_matrix = torch.cat(
                [
                    demographic,
                    not_yet_sliced[start:end],
                ]
            )
            # Handling labels for truncated inputs
            if start != self.n_demographic_rows:
                # Make the label at self.n_demographic_rows (slicing start point) ignored
                # NOTE: During autoregressive inference, this position is supposed to be a timedelta within a determined span.
                #       Anything else should not be positioned here.
                processed_matrix[self.n_demographic_rows, -1] = (
                    config.LOGITS_IGNORE_INDEX
                )

            # Handling the label for the last row
            if end < timeline_length:
                # NOTE: This label is added to the first row, and this will be rolled back to the end later.
                # WARNING: Do NOT try to sample the last label from `timeline_and_labels`. It is not shuffled.
                last_label = not_yet_sliced[end, -1].item()
                processed_matrix[0, -1] = last_label
            else:
                processed_matrix[0, -1] = config.LOGITS_IGNORE_INDEX

        # Case 3. short timelines
        else:
            # Pad the timeline
            pad_length = self.max_sequence_length - timeline_length
            pad = self.padding_template[:pad_length, :]
            processed_matrix = torch.cat([not_yet_sliced, pad])
            start = self.n_demographic_rows
            end = self.max_sequence_length
            n_sampled = 1

        # Convert objects
        n_sampled = torch.tensor(
            n_sampled, dtype=torch.float32, requires_grad=False
        ).reshape(1)

        return processed_matrix, start, end, n_sampled

    def _roll_labels(self, processed_matrix: torch.Tensor) -> torch.Tensor:
        """Shifts labels by one for loss calculations"""
        processed_matrix[:, -1] = torch.roll(processed_matrix[:, -1], -1)
        return processed_matrix

    def _preprocess(self, data: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a series of preprocess steps.
        Override this method and other submethods seen in this method if you inherit this class
        to define a subclass.
        Args:
            data (list): Data bundle that contains the followings in the order below.
                - timeline_and_labels (torch.Tensor): Input matrix concatenated with a series of labels.
                - slice_points (list[int]): List of indexes where slicing can be started.
                - shuffle_index_candidates (list[list[int]]): List of shuffled index series.
                    Because shuffling the indexes while preserving the chronological order takes some time,
                    some sets of shuffled indexes are prepared. One of them is randomly chosen for shuffling.
                - timedelta_series_pair (pd.DataFrame): Pair of timedelta series. The first column is the original
                    timedelta series, and the second is a shifted timedelta series.
                - catalog_indexes (list): This is currently not used during the training.
        Returns:
            processed_matrix (torch.Tensor): Processed matrix.
            n_sampled (torch.Tensor): Number of resampling.
        """
        # Unpack data
        (
            timeline_and_labels,
            slice_points,
            shuffle_index_candidates,
            timedelta_series_pair,
            catalog_indexes,
        ) = data
        # Shuffle and slice
        processed_matrix, slice_start, slice_end, n_sampled = self._shuffle_and_slice(
            timeline_and_labels,
            slice_points,
            shuffle_index_candidates,
        )
        # ***** < Start of data augmentation > *****
        if self.train:
            # No.1: add noise to timedelta
            processed_matrix = self._add_timedelta_noise(
                timedelta_series_pair, processed_matrix, slice_start, slice_end
            )
        # ***** < End of data augmentation > *****

        # Shift labels at the end (For next-token prediction loss criteria)
        processed_matrix = self._roll_labels(processed_matrix)

        return processed_matrix, n_sampled

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index) -> torch.Tensor:
        file = self.resampled_files[index]
        with open(file, "rb") as f:
            data = pickle.load(f)
        return self._preprocess(data)

    # *******************************
    # * Data augmentation functions *
    # *******************************
    def _add_timedelta_noise(
        self,
        timedelta_series_pair: pd.DataFrame,
        processed_matrix: torch.Tensor,
        start: int,
        end: int,
        min_noise: int = -60,
        max_noise: int = 60,
    ) -> torch.Tensor:
        """Adds random noise to timedelta.
        Because adding noise can result in changes in labels, updated labels are created here.
        Args:
            timedelta_series_pair (pd.DataFrame): Pair of timedelta series needed for data augmentation.
                This is a part of the training timeline data bundle.
            processed_matrix (torch.Tensor): Matrix to be processed.
            start (int): Starting index for slicing the dataframe (timedelta_series_pair).
            end (int): Ending index for slicing the dataframe (timedelta_series_pair).
                These two indexes (start, end) are the index used to slice out the original timeline.
            min_noise (int = -60): Smallest value of the noise. The unit is 'minute'.
            max_noise (int = 60): Largest value of the noise. The unit is 'minute'.
        Returns:
            processed_matrix (torch.Tensor): Processed matrix. Noise is added to timedelta values, and
                labels are properly updated.
        """
        # Trim timedelta series
        # NOTE: Sliced one row ahead to catch the label for the final row of the timeline.
        #       Pandas does not raise an error if the slice end is out of index.
        # NOTE: timedelta_series_pair does not contain paddings.
        sliced_td_pair = pd.concat(
            [
                # Pad for demographics
                timedelta_series_pair.iloc[0 : self.n_demographic_rows, :],
                # Actual sliced rows
                timedelta_series_pair.iloc[start : end + 1, :],
            ],
            axis=0,
        )
        # NOTE: Don't use 'sliced_td_pair.loc[:, "original"]' in the line below, because it inadvertently includes the first row (demographic).
        td_indexes = sliced_td_pair.loc[:, "shifted"].notna().values.nonzero()[0]
        min_td = sliced_td_pair.iloc[0, 0]
        # Add random noise
        random_int = random.randint(min_noise, max_noise)
        noise = timedelta(minutes=random_int)
        sliced_td_pair.iloc[self.n_demographic_rows :, :] += noise
        sliced_td_pair = sliced_td_pair.clip(lower=min_td)
        # Create a new input matrix and labels
        new_td_matrix = preprocess_timedelta_series(sliced_td_pair["original"])
        new_td_labels = create_labels_from_timedelta_pairs(
            sliced_td_pair["original"],
            sliced_td_pair["shifted"],
            first_timedelta_label=self.first_timedelta_label,
            small_step=self.td_small_step,
            large_step=self.td_large_step,
        )
        new_td_matrix = torch.tensor(new_td_matrix.values).float()
        new_td_labels = torch.tensor(new_td_labels.values).float()

        # <-- Handling of extra one row ahead -->
        # Replace old values with new ones
        if td_indexes[-1] == self.max_sequence_length:
            # Place the final label at the top so that it can be rolled back to the final row.
            processed_matrix[0, -1] = new_td_labels[-1]
            # Trim out the final index
            td_indexes = td_indexes[:-1]
        # <------------------------------------->

        # Replace input with new values
        processed_matrix[td_indexes, : self.timedelta_dim] = new_td_matrix[
            td_indexes, :
        ]
        processed_matrix[td_indexes, -1] = new_td_labels[td_indexes]

        return processed_matrix
