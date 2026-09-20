"""
Differential privacy mechanism used by PPDOE.
Paper states ε = 0.1. Optional CUDA path for noise generation.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from .device import get_device


class DifferentialPrivacy:
    """
    Basic Gaussian mechanism for DP.
    Privacy unit defaults to client (common in FL).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        delta: float = 1e-5,
        clipping_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_norm = clipping_norm
        if noise_multiplier is None:
            self.noise_multiplier = max(0.5, 1.0 / max(epsilon, 1e-6))
        else:
            self.noise_multiplier = noise_multiplier
        self.rng = np.random.RandomState(seed)
        self.spent_epsilon = 0.0
        self.rounds = 0
        self.device = device

    def clip(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > self.clipping_norm:
            return vector * (self.clipping_norm / norm)
        return vector

    def add_noise(self, vector: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise; uses CUDA when available."""
        clipped = self.clip(vector.astype(np.float32))
        noise_std = self.noise_multiplier * self.clipping_norm
        dev = self.device or get_device()
        if dev == "cuda":
            try:
                import torch
                t = torch.from_numpy(clipped).to("cuda")
                noise = torch.randn_like(t) * noise_std
                out = (t + noise).cpu().numpy().astype(np.float32)
                self.rounds += 1
                self.spent_epsilon += self.epsilon
                return out
            except Exception:
                pass
        noise = self.rng.normal(0.0, noise_std, size=clipped.shape).astype(np.float32)
        self.rounds += 1
        self.spent_epsilon += self.epsilon
        return clipped + noise

    def privatize_update(self, update: np.ndarray) -> np.ndarray:
        return self.add_noise(update)

    def privatize_prototypes(self, prototypes: np.ndarray) -> np.ndarray:
        out = np.zeros_like(prototypes)
        for i in range(prototypes.shape[0]):
            out[i] = self.add_noise(prototypes[i])
        return out

    def current_budget(self) -> Tuple[float, float]:
        return self.spent_epsilon, self.delta

    def reset(self) -> None:
        self.spent_epsilon = 0.0
        self.rounds = 0
