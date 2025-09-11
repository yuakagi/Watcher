"""Encoding layers"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class TimedeltaEncoding(nn.Module):
    """Layer to encode timedelta values."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        n_ffn: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, embedding_dim))
        layers.append(nn.ReLU())
        # Additional layers if n_ffn > 1
        for _ in range(n_ffn - 1):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.ReLU())
        # Removing the last ReLU
        self.linear = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor with timedelta values.
        """
        # Create a mask of the nan values in the input.
        nan_mask = torch.isnan(x[:, :, 0:1])
        # Compute the outputs
        out = self.linear(torch.nan_to_num(x))
        # Set the outputs to zero where there was a nan value in the input.
        out.masked_fill_(nan_mask, 0.0)

        return out


class NumericEncoding(nn.Module):
    """Layer to encode numeric values."""

    def __init__(
        self,
        embedding_dim: int,
        n_ffn: int = 2,
        input_dim: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, embedding_dim))
        layers.append(nn.ReLU())
        # Additional layers if n_ffn > 1
        for _ in range(n_ffn - 1):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.ReLU())
        # Removing the last ReLU
        self.linear = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with numeric values.
        """
        # Create a mask of the nan values in the input.
        nan_mask = torch.isnan(x[:, :, 0:1])
        # Compute the outputs
        out = torch.nan_to_num(x)
        out = self.linear(out)
        # Set the outputs to zero where there was a nan value in the input.
        out.masked_fill_(nan_mask, 0.0)
        return out


class SummingCategoricalEmbedding(nn.Module):
    """Layer to encode categorical inputs.
    Normalization is not performed inside this module. Make sure to properly normalize the outputs
    outside this module.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()

        # Ensure the num_embeddings is a multiple of 64
        num_embeddings = 64 * math.ceil(vocab_size / 64)
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode="sum",
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with categorical values.
        """
        batch_size, sequence_length, input_dim = x.size()
        # EmbeddingBag only supports 2D tensors, therefore the input needs reshaping.
        out = self.embedding(x.view(-1, input_dim))
        out = out.view(batch_size, sequence_length, -1)
        return out


class WatcherEncoder(nn.Module):
    def __init__(
        self,
        max_sequence_length: int,
        vocab_size: int,
        embedding_dim: int,
        categorical_dim: int,
        timedelta_dim: int = 5,
        numeric_dim: int = 1,
        padding_idx: int = 0,
        n_ffn: int = 2,
        epsilon: float = 1e-10,
    ):
        super().__init__()
        self.numeric_start = timedelta_dim
        self.categorical_start = self.numeric_start + numeric_dim
        self.adm_status_start = self.categorical_start + categorical_dim
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.categorical_embed = SummingCategoricalEmbedding(
            vocab_size, embedding_dim, padding_idx
        )
        self.numeric_encoding = NumericEncoding(
            embedding_dim=embedding_dim,
            n_ffn=n_ffn,
            input_dim=numeric_dim,
        )
        self.timedelta_encoding = TimedeltaEncoding(
            embedding_dim=embedding_dim,
            input_dim=timedelta_dim,
            n_ffn=n_ffn,
        )
        self.admission_encoding = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(1, 1, embedding_dim))
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.epsilon = epsilon

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes input"""
        # Split the input by input data types.
        td, num, categ, adm_status = torch.tensor_split(
            x,
            [self.numeric_start, self.categorical_start, self.adm_status_start],
            dim=2,
        )
        # Encode inputs. Note that, among categorical, numeric and timedelta encoded vectors, only one of them categorical is non-zero and others are zero vectors.
        encoded = self.categorical_embed(categ.long())
        encoded += self.numeric_encoding(num)
        encoded += self.timedelta_encoding(td)
        encoded = F.normalize(encoded, p=2, dim=2, eps=self.epsilon)  # L2 norm
        # Add admission status encoding
        encoded += adm_status * self.admission_encoding
        # Layer norm
        # TODO (Yu Akagi): Double normalization may be redundant
        encoded = self.norm(encoded)

        return encoded
