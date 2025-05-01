"""Utils for model inference"""

from datetime import timedelta
import torch
import pandas as pd
import numpy as np
from ..general_params import watcher_config as config


def extract_timedelta_sequence(
    timelines: list[torch.Tensor] | torch.Tensor, flatten: bool = False
) -> list[list[timedelta]] | list[timedelta]:
    """Creates series of timedelta from given timelines

    Args:
        timelines (list): List of 3D tensors. You can provide a timeline tensor (3D).
        flatten (bool, optional): If true, the returned list is flattened.
    Returns:
        timedelta_sequence (list): List of lists, with each sublist contains a series of timedelta objects.
            Each list is just as long as its corresponding timeline tensor. positions for values other than timedelta
            are filled by the latest timedelta values.
            If 'flatten' is true, then a flattened list of timedelta is returned.
    """
    # Initialize
    timedelta_dim = len(config.TIMEDELTA_COMPONENT_COLS)
    if isinstance(timelines, torch.Tensor):
        timelines = [t.unsqueeze(0) for t in timelines]
    # Get a series of timeline lengths
    lengths = [t.size(1) for t in timelines]
    # Concatenate all timelines
    flattened_trj = torch.cat(timelines, dim=1).clone().float().cpu().numpy()
    # Slice out columns for timedelta from the input matrix
    timedelta_matrix = flattened_trj[0, :, :timedelta_dim]
    # Convert normalized values to the original scale of days and minutes
    timedelta_matrix[:, 0] = timedelta_matrix[:, 0] * 365 * 120  # years->days
    timedelta_matrix[:, 1] = timedelta_matrix[:, 1] * 30 * 12  # months->days
    timedelta_matrix[:, 2] = timedelta_matrix[:, 2] * 29  # days
    timedelta_matrix[:, 3] = timedelta_matrix[:, 3] * 60 * 23  # hours->minutes
    timedelta_matrix[:, 4] = timedelta_matrix[:, 4] * 59  # minutes

    # Handling arithmetic underflow
    timedelta_matrix = np.round(timedelta_matrix)
    # Create a dataframe for timedelta operations
    timedelta_df = pd.DataFrame(timedelta_matrix, columns=["y", "m", "d", "h", "M"])
    # Eliminate Nan by ffill()
    timedelta_df = timedelta_df.ffill()
    # Convert values to timedelta objects
    for col in ["y", "m", "d"]:
        timedelta_df[col] = pd.to_timedelta(timedelta_df[col], unit="d")
    for col in ["h", "M"]:
        timedelta_df[col] = pd.to_timedelta(timedelta_df[col], unit="min")
    # Add timedelta objects
    timedelta_df["sum"] = timedelta_df.sum(axis=1, skipna=False)
    # Finalize
    summed = timedelta_df["sum"].to_list()

    if flatten:
        timedelta_sequences = summed
    else:
        timedelta_sequences = []
        end = 0
        for length in lengths:
            start = end
            end += length
            timedelta_seq = summed[start:end]
            timedelta_sequences.append(timedelta_seq)

    return timedelta_sequences
