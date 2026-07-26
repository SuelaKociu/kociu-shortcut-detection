"""Label-conditioned and label-free representation statistics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class MetricProfile:
    tss: float
    spectral_concentration: float
    local_sensitivity: float
    def as_array(self) -> np.ndarray:
        return np.asarray([self.tss, self.spectral_concentration, self.local_sensitivity], dtype=np.float64)

def _zscore_columns(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError("H must be a two-dimensional sample-by-feature matrix")
    scale = H.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (H - H.mean(axis=0)) / scale

def task_structured_separability(H: np.ndarray, labels: np.ndarray) -> float:
    H = _zscore_columns(H)
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != len(H):
        raise ValueError("labels and H must contain the same number of samples")
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("TSS currently requires exactly two classes")
    c0, c1 = H[labels == classes[0]], H[labels == classes[1]]
    if min(len(c0), len(c1)) < 2:
        raise ValueError("each class must contain at least two samples")
    m0, m1 = c0.mean(axis=0), c1.mean(axis=0)
    within = 0.5 * (np.linalg.norm(c0 - m0, axis=1).mean() + np.linalg.norm(c1 - m1, axis=1).mean())
    between = float(np.linalg.norm(m0 - m1))
    denom = within + between
    return 0.0 if denom < 1e-12 else float(between / denom)

def spectral_concentration(H: np.ndarray) -> float:
    H = _zscore_columns(H)
    if H.shape[1] < 2:
        raise ValueError("spectral concentration requires at least two hidden dimensions")
    corr = np.nan_to_num(np.corrcoef(H.T), nan=0.0, posinf=0.0, neginf=0.0)
    eig = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    total = eig.sum()
    if total < 1e-12:
        return 0.0
    p = eig / total
    entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p))
    return float(np.clip(1.0 - entropy, 0.0, 1.0))

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def local_sensitivity(hidden_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, *, n_perturbations: int = 50, sigma: float = 0.05, seed: int = 0) -> float:
    if n_perturbations <= 0 or sigma <= 0:
        raise ValueError("n_perturbations and sigma must be positive")
    X = np.asarray(X, dtype=np.float32)
    base = np.asarray(hidden_fn(X), dtype=np.float64)
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_perturbations):
        perturbed = X + rng.randn(*X.shape).astype(np.float32) * sigma
        changed = np.asarray(hidden_fn(perturbed), dtype=np.float64)
        values.append(_pearson(np.linalg.norm(perturbed - X, axis=1), np.linalg.norm(changed - base, axis=1)))
    return float(np.mean(values))

def profile(hidden_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, labels: np.ndarray, *, n_perturbations: int = 50, sigma: float = 0.05, seed: int = 0) -> MetricProfile:
    H = hidden_fn(X)
    return MetricProfile(task_structured_separability(H, labels), spectral_concentration(H), local_sensitivity(hidden_fn, X, n_perturbations=n_perturbations, sigma=sigma, seed=seed))
