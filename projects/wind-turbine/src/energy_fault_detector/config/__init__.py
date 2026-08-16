"""Configuration package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "config")

from .config import Config
from .base_config import InvalidConfigFile
from .quickstart_config import generate_quickstart_config

__all__ = ["Config", "InvalidConfigFile", "generate_quickstart_config"]
