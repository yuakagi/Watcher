"""Layers for normalization"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm
    Args:
        embed_dim (int): Model's embedding dimension.
        eps (float): Value to avoid ZeroDivisionError.
    """

    def __init__(self, embed_dim: int, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            out: Normalized tensor.
        """
        # Cast to float32 for precision
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.epsilon)
        # Cast back to original dtype
        out = out.type_as(x)
        out = out * self.scale
        return out
