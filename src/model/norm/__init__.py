"""Normalization modules and factory.

Groups the custom normalization layers (``DynamicTanhNorm``, ``Derf``) and the
``get_norm`` factory function that selects one from the model configuration.
"""

from __future__ import annotations

from .derf import Derf
from .dynamic_tanh import DynamicTanhNorm
from .factory import get_norm

__all__ = [
    "DynamicTanhNorm",
    "Derf",
    "get_norm",
]
