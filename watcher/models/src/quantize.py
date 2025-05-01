import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.types


def quantize_per_channel(
    x: Tensor,
    dtype: torch.dtype = torch.int8,
    quant_min: int = -128,
    quant_max: int = 127,
    affine: bool = False,
):
    """
    Quantizes model parameter weights (e.g., `nn.Linear` weights) per channel.

    This function performs either symmetric or affine (asymmetric) quantization
    of the input tensor along its second dimension (channels). It is designed
    to reduce the memory footprint of model parameters, enabling efficient
    storage and inference in low-precision formats such as `int8`.

    Args:
        x (torch.Tensor):
            Input tensor to be quantized, typically parameter weights from
            layers like `nn.Linear`. Must have at least two dimensions.
        quant_min (int):
            Minimum value for the quantized range (e.g., `-128` for `int8` or `0` for `uint8`).
        quant_max (int):
            Maximum value for the quantized range (e.g., `127` for `int8` or `255` for `uint8`).
        dtype (torch.dtype):
            Target quantized data type (e.g., `torch.int8` or `torch.uint8`). Default is `torch.int8`.
        affine (bool, optional):
            If `True`, performs affine (asymmetric) quantization, where a non-zero
            zero point is used to map the floating-point range to the quantized range.
            If `False`, performs symmetric quantization, where the quantized range is
            centered around zero. Defaults to `False`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - **Quantized Tensor**: The input tensor quantized to the specified dtype,
              with the same shape as the input.
            - **Scales Tensor**: A per-channel scale tensor (one value per channel)
              used to dequantize the quantized tensor.
            - **Zero Points Tensor**: A per-channel zero point tensor (one value per channel),
              indicating the offset used during quantization. For symmetric quantization,
              this will always be zero.

    Example:
        Quantizing weights for an `nn.Linear` layer:
        >>> weights = torch.randn(128, 256)  # Example weights
        >>> quant_min, quant_max = -128, 127
        >>> dtype = torch.int8
        >>> quant_weights, scales, zero_points = quantize_per_channel(
        ...     weights, dtype=dtype, quant_min=quant_min, quant_max=quant_max, affine=True
        ... )
        >>> print(f"Quantized Weights: {quant_weights.shape}")
        >>> print(f"Scales: {scales.shape}")
        >>> print(f"Zero Points: {zero_points.shape}")

    Notes:
        - For affine quantization (`affine=True`), this function calculates per-channel
          scales and zero points using both the minimum and maximum values of each channel.
        - For symmetric quantization (`affine=False`), the scale is calculated based on the
          maximum absolute value of each channel, and the zero point is always set to zero.
        - The function assumes that the second dimension of the input tensor represents
          the channels to be quantized (e.g., for `nn.Linear`, this is typically the output features).
    """
    # Set up
    eps = torch.finfo(torch.float32).eps

    with torch.device(x.device):
        # 1. Get min and max values in the input `x`
        min_val, max_val = torch.aminmax(x, dim=1)

        # 2. Compute scales and zero-points
        if affine:
            # For affine quantization, consider both min and max for scaling
            scales = (max_val - min_val) / (quant_max - quant_min)
            scales = torch.clamp(scales, min=eps).to(x.dtype)
            zero_points = torch.round(quant_min - min_val / scales).to(torch.int64)
            zero_points = torch.clamp(zero_points, quant_min, quant_max).to(torch.int64)
        else:
            # For symmetric quantization, use the maximum absolute value
            abs_max_val = torch.max(torch.abs(min_val), torch.abs(max_val))
            scales = abs_max_val / ((quant_max - quant_min) / 2)
            scales = torch.clamp(scales, min=eps).to(x.dtype)
            zero_points = torch.zeros_like(min_val, dtype=torch.int64)

        # 3. Quantize the input
        x_quant = x / scales.unsqueeze(-1)
        x_quant = torch.round(x_quant)
        x_quant = x_quant + zero_points.unsqueeze(-1)

        # 4. Clamp weights
        x_quant = torch.clamp(x_quant, quant_min, quant_max).to(dtype)

    return x_quant, scales, zero_points


def quantize_linear_weights(mod: nn.Module):
    """
    Applies weight quantization to nn.Linear layers in a model.

    Args:
        mod (nn.Module): Input PyTorch model.

    Returns:
        nn.Module: Model with quantized weights.
    """
    for name, child in mod.named_children():
        if isinstance(child, nn.Linear):
            ch_dtype = child.weight.dtype
            ch_device = child.weight.device
            use_bias = child.bias is not None
            int8_weight, scales, _ = quantize_per_channel(
                child.weight.float(), torch.int8, -128, 127
            )
            # Replace the module
            new_layer = Int8WeightLinear(
                child.in_features,
                child.out_features,
                bias=use_bias,
                dtype=ch_dtype,
                device=ch_device,
            )
            new_layer.weight = int8_weight
            new_layer.scales = scales.unsqueeze(-1).to(ch_dtype)
            if use_bias:
                new_layer.bias = child.bias.clone()
            setattr(mod, name, new_layer)
        else:
            quantize_linear_weights(child)

    return mod


class Int8WeightLinear(torch.nn.Module):
    """
    Linear layer with int8 weights for faster inference.

    Args:
        in_features (int): Input feature size.
        out_features (int): Output feature size.
        bias (bool, optional): Include bias if True. Defaults to True.
        device (torch.device, optional): Device to store the tensors.
        dtype (torch.dtype, optional): Data type for weights and scales.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If True, include a bias term in the linear transformation. Defaults to True.
            device (torch.device, optional): Device to store the tensors (e.g., 'cpu' or 'cuda').
            dtype (torch.dtype, optional): Data type for weights and scales (e.g., torch.float32).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.core_dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones((out_features, 1), **factory_kwargs))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, **factory_kwargs))
        else:
            self.register_buffer("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        dequantized_weight = self.weight.to(dtype=self.core_dtype) * self.scales
        # pylint: disable=not-callable
        output = F.linear(input, dequantized_weight, self.bias)
        return output
