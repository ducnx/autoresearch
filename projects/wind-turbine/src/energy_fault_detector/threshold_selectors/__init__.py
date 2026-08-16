"""Threshold selector package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "threshold_selectors")

from .fdr_threshold import FDRSelector
from .fbeta_threshold import FbetaSelector
from .quantile_threshold import QuantileThresholdSelector
from .adaptive_threshold import AdaptiveThresholdSelector

__all__ = ["FDRSelector", "FbetaSelector", "QuantileThresholdSelector", "AdaptiveThresholdSelector"]
