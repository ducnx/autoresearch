"""Autoencoder model package wrapper."""

from energy_fault_detector._namespace import extend_path

extend_path(globals(), "models")

from .multilayer_autoencoder import MultilayerAutoencoder
from .conditional_autoencoder import ConditionalAE
from .vrae import VRAE

__all__ = ["MultilayerAutoencoder", "ConditionalAE", "VRAE"]
