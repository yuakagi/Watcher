from scipy.special import logit, expit
import numpy as np
from ..general_params import watcher_config as config


def compute_numeric_steps(num_bins: int):
    """Computes numeric steps."""
    num_steps = np.linspace(0, 1, num_bins)
    return num_steps


def discretize_percentiles(num_bins: int, eps: float = 1e-7) -> np.ndarray:
    """
    Discretizes the percentile space [0, 1].

    A uniform grid is applied in logit space to create finer resolution near 0 and 1,
    which helps capture long-tailed distributions. The logit grid is transformed back to [0, 1]
    using the sigmoid function.
    Each resulting bin is assigned a uniformly spaced numeric label in [0.0, 1.0].

    Args:
        num_bins (int): Number of bins to divide the percentile space.
        eps (float): Small offset to avoid logit(0) and logit(1). Default is 1e-7.

    Returns:
        percentile_steps (np.ndarray): Non-uniform percentile bin edges with finer resolution near 0 and 1.
    """
    # Uniformly spaced values in logit space → transformed to percentile space
    logit_space = np.linspace(logit(eps), logit(1 - eps), num_bins)
    percentile_steps = expit(logit_space)
    return percentile_steps


def compute_zscore_centers(num_bins: int) -> np.ndarray:
    """Computes center values of z-score bins.

    The last bin center is capped at NUM_UPPER_BOUND.
    """
    step_size = (config.NUM_UPPER_BOUND - config.NUM_LOWER_BOUND) / (num_bins - 1)
    half_step = step_size / 2
    edges = np.linspace(config.NUM_LOWER_BOUND, config.NUM_UPPER_BOUND, num_bins)
    centers = np.minimum(edges + half_step, config.NUM_UPPER_BOUND)
    return centers
