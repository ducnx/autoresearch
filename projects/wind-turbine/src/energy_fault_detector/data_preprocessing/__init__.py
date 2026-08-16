"""Data preprocessing package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "data_preprocessing")

from .data_preprocessor import DataPreprocessor
from .data_clipper import DataClipper
from .data_splitter import BlockDataSplitter

__all__ = ["DataPreprocessor", "DataClipper", "BlockDataSplitter"]
