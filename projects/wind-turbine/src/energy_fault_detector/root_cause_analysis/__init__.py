"""Root-cause analysis package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "root_cause_analysis")

from .arcana import Arcana

__all__ = ["Arcana"]
