"""Energy fault detector package for wind-turbine experiments."""

from energy_fault_detector.registration import register_defaults, registry

register_defaults()

from energy_fault_detector.config import Config, generate_quickstart_config
from energy_fault_detector.fault_detector import FaultDetector

__all__ = ["Config", "FaultDetector", "generate_quickstart_config", "registry"]
