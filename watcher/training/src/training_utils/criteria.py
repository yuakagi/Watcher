import torch
import torch.nn as nn
import torch.nn.functional as F


class WatcherLoss(nn.Module):
    def __init__(
        self,
        ignore_index,
        gamma: float,
        scaled: bool,
    ):
        """Custom loss for Watcher.

        Args:
            ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            gamma (float): Focusing parameter for Focal Loss. Defaults to Cross-Entropy Loss if gamma == 0.
            scaled (bool): If True, loss is scaled by the number of resamplings.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal = gamma > 0
        self.gamma = gamma
        self.scaled = scaled
        self.log_prob_indexes = None

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, scaling_factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross-entropy loss with optional focal loss and scaling.

        Args:
            input (torch.Tensor): Logits of shape (B, S, C).
            target (torch.Tensor): Target class indices of shape (B, S).
            scaling_factors (torch.Tensor): Scaling factors of shape (B,) or (B, 1).
                If self.scaled is False, scaling factors are ignored.

        Returns:
            torch.Tensor: Scaled and masked cross-entropy or focal loss.
        """
        # Reshape input and target for per-token loss computation
        b_size, seq_len, n_class = input.shape
        input_flat = input.view(-1, n_class)
        target_flat = target.view(-1)

        # Compute log probabilities
        log_probs = F.log_softmax(input_flat, dim=-1)

        # Set up slicing indexes
        if (self.log_prob_indexes is None) or log_probs.size(
            0
        ) != self.log_prob_indexes.size(0):
            self.log_prob_indexes = torch.arange(
                log_probs.size(0), device=log_probs.device
            )

        # Filter out ignored indices before computing loss
        valid_indices = target_flat != self.ignore_index
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        target_log_probs = log_probs[
            self.log_prob_indexes[valid_indices], target_flat[valid_indices]
        ]

        # Compute focal loss if applicable
        if self.focal:
            pt = target_log_probs.exp()
            loss = -((1 - pt) ** self.gamma) * target_log_probs
        else:
            loss = -target_log_probs  # Cross-Entropy Loss

        # Apply scaling factors if enabled
        if self.scaled:
            assert (
                scaling_factors is not None
            ), "Scaling factors must be provided when scaled=True"
            scaling_factors = scaling_factors.float().view(b_size, 1)
            sf_flat = scaling_factors.repeat(1, seq_len).view(-1)
            loss = loss / sf_flat[valid_indices]

        return loss.mean()
