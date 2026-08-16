"""Quick fault detection package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "quick_fault_detection")

from .quick_fault_detector import quick_fault_detector

__all__ = ["quick_fault_detector"]
