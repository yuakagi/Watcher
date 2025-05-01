"""The transformer layers"""

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, embed_dim: int, num_heads: int, max_sequence_length: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of heads."
            )
        self.head_dim = int(embed_dim / num_heads)
        self.scale_factor = 1 / (self.head_dim**0.5)
        self.qkv_projection = nn.Linear(embed_dim, 3 * embed_dim)
        self.resid_proj = nn.Linear(embed_dim, embed_dim)
        # Initialize buffers
        self.register_buffer(
            "attn_mask",
            torch.tril(
                torch.ones(
                    1,
                    1,
                    self.max_sequence_length,
                    self.max_sequence_length,
                    dtype=torch.bool,
                ),
                diagonal=0,
            ),
            persistent=False,
        )
        self.register_buffer(
            "positions", torch.arange(0, self.max_sequence_length), persistent=False
        )
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.register_buffer("attention_weights", None, persistent=False)

    @property
    def device(self):
        """Returns the current device."""
        return self.attn_mask.device

    def setup_cache(self, batch_size: int, dtype: torch.dtype):
        """Initializes cache."""
        with torch.device(self.device):
            cache_cap = (
                batch_size,
                self.num_heads,
                self.max_sequence_length,
                self.head_dim,
            )
            k_cache = torch.zeros(cache_cap).to(dtype)
            self.register_buffer("k_cache", k_cache, persistent=False)
            self.register_buffer("v_cache", k_cache.clone(), persistent=False)

    def delete_cache(self):
        """Deletes cache."""
        # pylint: disable=no-member
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.positions[:] = self.positions - self.positions[0]

    def trim_cache(self, actives: torch.Tensor, new_size: int):
        """Trims KV cache."""
        # pylint: disable=access-member-before-definition
        self.k_cache[: actives.size(0)] = self.k_cache[actives]
        self.v_cache[: actives.size(0)] = self.v_cache[actives]
        # pylint: disable=attribute-defined-outside-init
        self.k_cache = self.k_cache[:new_size]
        self.v_cache = self.v_cache[:new_size]

    def empty_cache(self) -> None:
        """Clears KV cache."""
        # pylint: disable=no-member
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.positions[:] = self.positions - self.positions[0]

    def update_cache(
        self, new_k: torch.Tensor, new_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates cache."""
        length = new_k.size(2)
        # Update cache
        # pylint: disable=no-member
        self.k_cache[:, :, self.positions[:length]] = new_k
        self.v_cache[:, :, self.positions[:length]] = new_v
        # Slice attention mask
        mask = self.attn_mask[:, :, self.positions[:length]]
        # Shift positions
        self.positions += length
        # K and V
        k = self.k_cache
        v = self.v_cache
        return k, v, mask

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        record_attention_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor to split.
            use_cache (bool): If true, KV-cache is used.
            record_attention_weights (bool): If true, attention weights are stored.
                The weights are saved as self.attention_weights.
        Returns:
            out (torch.Tensor): Split input.
        """
        # NOTE: F.sdpa returns unexpected results if not using float16 or bfloat16.
        #       Ensure to use bfloat16.

        # Generate Q, K, V and split them into multiple heads
        batch_size = x.size(0)
        out = self.qkv_projection(x)
        out = out.view(batch_size, -1, self.num_heads, 3 * self.head_dim).permute(
            0, 2, 1, 3
        )
        q, new_k, new_v = torch.tensor_split(out, 3, dim=-1)
        # Case 1. inference with KV-cache
        if use_cache:
            # Compute K, V and attention mask
            k, v, mask = self.update_cache(new_k, new_v)
            # Disable the default causal mask
            is_causal = False

        # Case 2. Inference without cache (for training)
        else:
            k, v = new_k, new_v
            is_causal = True
            mask = None

        # SDPA pylint: disable=not-callable
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=mask,
        )
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.resid_proj(out)

        # Record attention weights for attention map visualization
        if record_attention_weights:
            qk = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale_factor
            if mask is None:
                temp_mask = self.attn_mask[:, :, : q.size(2), : k.size(2)]
            else:
                temp_mask = mask
            qk.masked_fill_(temp_mask.logical_not(), float("-inf"))
            qk = torch.softmax(qk, dim=-1)
            # False-positive related to 'register_buffer', pylint: disable=attribute-defined-outside-init
            self.attention_weights = torch.mean(qk[:, :, -1, :], dim=1).clone()

        return out


class TransformerBlock(nn.Module):
    """The Transformer block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout_rate: float,
        max_sequence_length: int,
    ):
        """Initializes the Transformer block."""
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            embed_dim, num_heads, max_sequence_length
        )
        # TODO: Consider changing to RMSNorm.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(embed_dim, ff_hidden_dim)),
                    ("gelu", nn.GELU()),
                    ("resid_proj", nn.Linear(ff_hidden_dim, embed_dim)),
                ]
            )
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        record_attention_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
            use_cache (bool): If true, KV-cache is used.
            record_attention_weights (bool): If true, attention weights are stored.
                The weights are saved as self.attention.attention_weights.
        Returns:
            x (torch.Tensor): Output tensor with shape (batch_size, sequence_length, embed_dim).
        """
        # ***** Norm + Attention *****
        x = x + self.dropout(
            self.attention(self.norm1(x), use_cache, record_attention_weights)
        )
        # +**** Feed-forward + Norm *****
        x = x + self.dropout(self.feed_forward(self.norm2(x)))

        return x


class TransformerLayerStack(nn.Module):
    """Transformer layer stack"""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout_rate: float,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ff_hidden_dim,
                    dropout_rate,
                    max_sequence_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.position_encoding = nn.Parameter(
            torch.normal(mean=0, std=0.01, size=(1, max_sequence_length, embed_dim))
        )
        # TODO: Consider using RMSNorm.
        self.norm = nn.LayerNorm(embed_dim)
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

    @property
    def cache_length(self) -> int | None:
        """Gets the current length of KV-cache."""
        positions = self.layers[0].attention.positions
        return positions[0].item()

    def setup_cache(self, batch_size: int, dtype: torch.dtype = torch.bfloat16):
        """Initialize cache."""
        for i in range(self.num_layers):
            self.layers[i].attention.setup_cache(batch_size, dtype)

    def delete_cache(self):
        """Deletes cache."""
        for i in range(self.num_layers):
            self.layers[i].attention.delete_cache()

    def trim_cache(self, actives: torch.Tensor, new_size: int):
        """Trims cache."""
        for i in range(self.num_layers):
            self.layers[i].attention.trim_cache(actives, new_size)

    def empty_cache(self):
        """Clears all cache."""
        for i in range(self.num_layers):
            self.layers[i].attention.empty_cache()

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        record_attention_weights: bool = False,
    ) -> torch.Tensor:
        """Forward pass for training.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, embed_dim).
            use_cache (bool): If true, KV-cache is used.
            record_attention_weights (bool): If true, attention weights are stored.
                The weights are saved in each module in self.layers.
        Returns:
            x: Output.
        """
        # ***** Positional encoding *****
        length = x.size(1)
        if use_cache:
            positions = self.layers[0].attention.positions
            pe = self.position_encoding[:, positions[:length]]
        else:
            pe = self.position_encoding[:, :length]
        out = x + pe

        # ***** Transformer blocks *****
        for layer in self.layers:
            out = layer(out, use_cache, record_attention_weights)
        # Final norm
        out = self.norm(out)

        return out
