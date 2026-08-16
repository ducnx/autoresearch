"""Core package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "core")

from .anomaly_score import AnomalyScore
from .autoencoder import Autoencoder
from .data_transformer import DataTransformer
from .threshold_selector import ThresholdSelector
from .fault_detection_result import FaultDetectionResult, ModelMetadata

__all__ = [
    "AnomalyScore",
    "Autoencoder",
    "DataTransformer",
    "ThresholdSelector",
    "FaultDetectionResult",
    "ModelMetadata",
]
