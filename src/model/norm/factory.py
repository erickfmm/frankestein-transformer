"""Normalization factory.

Selects the appropriate normalization module from the model configuration,
mirroring the optimizer factory pattern in ``src/model/optimizer/factory.py``.
"""

from __future__ import annotations

import torch.nn as nn

from .derf import Derf
from .dynamic_tanh import DynamicTanhNorm


def get_norm(config):
    """Factory function that returns a normalization module based on config.

    Selects among ``LayerNorm``, ``DynamicTanhNorm``, and ``Derf`` based on
    the ``norm_type`` field in the configuration.

    Note:
        ``rms_norm`` is intentionally NOT supported. See ``configs/schema.yaml``.

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
