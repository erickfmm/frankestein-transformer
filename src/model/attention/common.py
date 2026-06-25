"""Shared utilities for attention modules.

Provides quantization helpers (BitNet b1.58 ternary weights, 8-bit activations)
and the ``BitLinear`` layer.

Normalization layers (``DynamicTanhNorm``, ``Derf``) and the ``get_norm``
factory have been moved to ``src/model/norm/``.

References:
    Ma et al. (2024), "The Era of 1-bit LLMs: All Large Language Models are in
    1.58 Bits", arXiv:2402.17764.
"""

import torch.nn as nn
import torch.nn.functional as F


def activation_quant(x):
    """Quantize activations to 8-bit via straight-through estimator (STE).

    Scales activations to the range [-128, 127], rounds to integer, then
    rescales back. The STE passes gradients through the rounding operation
    unchanged, enabling end-to-end training with quantized activations.

    Args:
        x: Input tensor of any shape. The last dimension is used for
            per-token scaling.

    Returns:
        Tensor of same shape as ``x``, with values quantized to 8-bit
        precision. Gradients flow through the quantization via STE.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()  # STE


def weight_quant(w):
    """Quantize weights to ternary values {-1, 0, 1} via straight-through estimator.

    Scales weights by the inverse of their mean absolute value, rounds to the
    nearest integer in [-1, 1], then rescales. The STE passes gradients through
    the rounding operation unchanged.

    Args:
        w: Weight tensor of any shape.

    Returns:
        Tensor of same shape as ``w``, with values quantized to ternary
        precision. Gradients flow through the quantization via STE.
    """
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    w_quant = (w * scale).round().clamp(-1, 1) / scale
    return w + (w_quant - w).detach()  # STE


class BitLinear(nn.Linear):
    """BitNet b1.58 linear layer with ternary weight quantization.

    A drop-in replacement for ``nn.Linear`` that reduces VRAM usage by 3-4x
    through ternary weight quantization and 8-bit activation quantization.
    Applies LayerNorm to inputs before quantization for stability, as
    prescribed by the BitNet b1.58 formulation.

    Reference:
        Ma et al. (2024), "The Era of 1-bit LLMs", arXiv:2402.17764.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias to the output. Defaults to
            ``False``.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        """Forward pass with activation and weight quantization.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        x_norm = F.layer_norm(x, x.shape[1:])

        w_q = weight_quant(self.weight)
        x_q = activation_quant(x_norm)

        output = F.linear(x_q, w_q, self.bias)
        return output



