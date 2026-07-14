"""Normalization modules and factory.

Groups the custom normalization layers (``DynamicTanhNorm``, ``Derf``,
``RMSNorm``) and the ``get_norm`` factory function that selects one from the
model configuration.
"""

from __future__ import annotations

from .derf import Derf
from .dynamic_tanh import DynamicTanhNorm
from .factory import get_norm
from .rms import RMSNorm

__all__ = [
    "DynamicTanhNorm",
    "Derf",
    "RMSNorm",
    "get_norm",
]
