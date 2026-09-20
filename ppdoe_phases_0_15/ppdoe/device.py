"""
Device helper – optional CUDA acceleration.
Falls back to CPU when torch or a GPU is unavailable.
"""

from __future__ import annotations
from typing import Optional
import os

_DEVICE = None


def get_device(prefer: Optional[str] = None) -> str:
    """
    Return 'cuda' or 'cpu'.
    prefer: 'cuda' | 'cpu' | None (auto)
    Override with env PPDOE_DEVICE=cuda|cpu
    """
    global _DEVICE
    if _DEVICE is not None and prefer is None:
        return _DEVICE

    env = os.environ.get("PPDOE_DEVICE", "").lower()
    if prefer:
        choice = prefer.lower()
    elif env in ("cuda", "cpu"):
        choice = env
    else:
        choice = "cuda"  # try GPU first

    if choice == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                _DEVICE = "cuda"
                return _DEVICE
        except ImportError:
            pass
    _DEVICE = "cpu"
    return _DEVICE


def set_device(device: str) -> str:
    global _DEVICE
    _DEVICE = get_device(prefer=device)
    return _DEVICE


def has_cuda() -> bool:
    return get_device() == "cuda"
