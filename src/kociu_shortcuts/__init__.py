"""Internal-representation diagnostics for controlled shortcut-learning studies."""

from .data import Dataset, Standardizer, make_dataset
from .detector import paired_loso_predictions
from .metrics import MetricProfile, local_sensitivity, spectral_concentration, task_structured_separability

__all__ = [
    "Dataset",
    "MetricProfile",
    "Standardizer",
    "local_sensitivity",
    "make_dataset",
    "paired_loso_predictions",
    "spectral_concentration",
    "task_structured_separability",
]

__version__ = "2.1.0"
