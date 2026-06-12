"""Shared utilities for attention modules.

Provides quantization helpers (BitNet b1.58 ternary weights, 8-bit activations),
custom normalization layers (DynamicTanhNorm, Derf), and a factory function
for selecting the normalization type from the model configuration.

References:
    Ma et al. (2024), "The Era of 1-bit LLMs: All Large Language Models are in
    1.58 Bits", arXiv:2402.17764.
"""

import torch
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


class DynamicTanhNorm(nn.Module):
    """Dynamic Tanh normalization layer.

    Applies standard normalization (subtract mean, divide by standard deviation)
    followed by a learnable affine transformation through ``tanh``. The
    learnable parameters ``alpha`` and ``beta`` allow the layer to adapt the
    saturation point and slope of the tanh non-linearity per dimension.

    Args:
        dim: Number of features in the input (normalized dimension).
        eps: Small constant for numerical stability in standard deviation.
            Defaults to ``1e-6``.

    Attributes:
        alpha: Learnable scale parameter of shape ``(dim,)``, initialized to 1.
        beta: Learnable shift parameter of shape ``(dim,)``, initialized to 0.
        eps: Epsilon value for numerical stability.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        """Apply dynamic tanh normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape as ``x``, normalized and passed through
            ``tanh`` with learnable affine parameters.
        """
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        x_norm = (x - mu) / (sigma + self.eps)
        return torch.tanh(x_norm * self.alpha + self.beta)


class Derf(nn.Module):
    """Derf normalization layer.

    Applies the error function (erf) as a smooth, saturating non-linearity
    with learnable affine parameters. The formulation is::

        y = gamma * erf(alpha * x + s) + beta

    where ``alpha`` and ``s`` are scalar parameters controlling the slope and
    shift of the erf, and ``gamma``, ``beta`` are per-dimension scale and bias.

    Args:
        dim: Number of features in the input (normalized dimension).

    Attributes:
        alpha: Learnable scalar slope parameter, initialized to 1.0.
        s: Learnable scalar shift parameter, initialized to 0.0.
        gamma: Learnable per-dimension scale of shape ``(dim,)``, initialized
            to 1.
        beta: Learnable per-dimension bias of shape ``(dim,)``, initialized
            to 0.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.s = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Derf normalization.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Tensor of same shape as ``x``, transformed by the erf non-linearity
            with learnable affine parameters.
        """
        return self.gamma * torch.erf(self.alpha * x + self.s) + self.beta


def get_norm(config):
    """Factory function that returns a normalization module based on config.

    Selects among ``LayerNorm``, ``DynamicTanhNorm``, and ``Derf`` based on
    the ``norm_type`` field in the configuration.

    Args:
        config: Model configuration object with attributes ``norm_type``
            (one of ``"layer_norm"``, ``"dynamic_tanh"``, ``"derf"``) and
            ``hidden_size``.

    Returns:
        A normalization ``nn.Module`` instance appropriate for the requested
        ``norm_type``.

    Raises:
        AttributeError: If ``config`` does not have ``norm_type`` or
            ``hidden_size`` attributes.
    """
    if config.norm_type == "dynamic_tanh":
        return DynamicTanhNorm(config.hidden_size)
    if config.norm_type == "derf":
        return Derf(config.hidden_size)
    return nn.LayerNorm(config.hidden_size)
