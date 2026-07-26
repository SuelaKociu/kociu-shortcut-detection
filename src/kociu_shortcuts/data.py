"""Controlled synthetic data generators.

The confirmatory API keeps raw data generation separate from scaling. A single
reference standardizer can therefore be fitted once and applied to every member
of a paired comparison, preventing condition-specific preprocessing from
becoming an unintended detection cue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ShortcutMode = Literal[
    "linear", "signed_threshold", "product", "sparse_symmetric",
    "matched_control", "noise", "balanced",
]

@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    mode: str
    seed: int
    shortcut_mask: np.ndarray | None = None

@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=X.mean(axis=0), scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

def _core(n: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0 or n % 2:
        raise ValueError("n must be a positive even integer")
    n_per = n // 2
    class_zero = rng.randn(n_per, 2) + np.array([1.5, 1.5])
    class_one = rng.randn(n_per, 2) + np.array([-1.5, -1.5])
    X_core = np.concatenate([class_zero, class_one], axis=0)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(np.int64)
    return X_core, y

def make_dataset(n: int = 600, mode: ShortcutMode = "noise", seed: int = 0, *, shortcut_noise: float = 0.05, sparse_prevalence: float = 0.30, sparse_amplitude: float = 10.0) -> Dataset:
    if not 0.0 < sparse_prevalence < 1.0:
        raise ValueError("sparse_prevalence must lie strictly between 0 and 1")
    rng = np.random.RandomState(seed)
    X_core, y = _core(n, rng)
    shortcut_mask = None
    if mode == "linear":
        s1 = y + rng.randn(n) * shortcut_noise; s2 = rng.randn(n)
    elif mode == "signed_threshold":
        s1 = np.where(y == 1, rng.uniform(2.0, 3.0, n), rng.uniform(-3.0, -2.0, n)); s2 = rng.randn(n)
    elif mode == "product":
        s1 = rng.randn(n); s2 = np.where(y == 1, s1, -s1) + rng.randn(n) * 0.10
    elif mode == "sparse_symmetric":
        s1 = rng.randn(n); shortcut_mask = rng.rand(n) < sparse_prevalence
        signed_y = 2.0 * y - 1.0
        s1[shortcut_mask] = signed_y[shortcut_mask] * sparse_amplitude + rng.randn(shortcut_mask.sum()) * shortcut_noise
        s2 = rng.randn(n)
    elif mode == "matched_control":
        independent_binary = rng.permutation(y); s1 = independent_binary + rng.randn(n) * shortcut_noise; s2 = rng.randn(n)
    elif mode == "balanced":
        flip = rng.rand(n) < 0.5; s1 = np.where(flip, y, 1 - y) + rng.randn(n) * shortcut_noise; s2 = rng.randn(n)
    elif mode == "noise":
        s1 = rng.randn(n); s2 = rng.randn(n)
    else:
        raise ValueError(f"unknown shortcut mode: {mode}")
    X = np.column_stack([X_core, s1, s2]).astype(np.float32)
    return Dataset(X=X, y=y, mode=mode, seed=seed, shortcut_mask=shortcut_mask)
