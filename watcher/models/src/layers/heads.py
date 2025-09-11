"""Prediction heads"""

import math
from collections import OrderedDict
import torch
from torch import nn


class LogitsHead(nn.Module):
    """Head for producing logits"""

    def __init__(self, vocab_size: int, input_dim: int, ff_hidden_dim: int) -> None:
        super().__init__()
        # Ensure that the output size is a multiple of 64 for Tensor Core activation
        output_size = 64 * math.ceil(vocab_size / 64)
        self.vocab_size = vocab_size
        self.layer = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(input_dim, ff_hidden_dim)),
                    ("gelu", nn.GELU()),
                    ("vocab_proj", nn.Linear(ff_hidden_dim, output_size)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This forward pass trims out padded dims
        Args:
            x (Tensor): Input tensor.
        Returns:
            out (Tensor): Logits
        """
        out = self.layer(x)
        out = out[:, :, : self.vocab_size]
        return out
