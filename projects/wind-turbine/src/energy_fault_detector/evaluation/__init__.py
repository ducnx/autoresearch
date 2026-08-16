"""Evaluation package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "evaluation")

from .care_score import CAREScore
from .care2compare import Care2CompareDataset
from .predist_dataset import PreDistDataset

__all__ = ["CAREScore", "Care2CompareDataset", "PreDistDataset"]
