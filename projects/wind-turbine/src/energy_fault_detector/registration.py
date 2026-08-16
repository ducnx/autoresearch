"""Simple model registry used by the wind-turbine fault detector."""

from __future__ import annotations

from collections import defaultdict
from typing import Type


class Registry:
    def __init__(self) -> None:
        self._models: dict[str, dict[str, Type]] = defaultdict(dict)

    @staticmethod
    def _key(name: str) -> str:
        return str(name).replace("-", "_").lower()

    def register(self, model_type: str, name: str, cls: Type) -> None:
        self._models[model_type][self._key(name)] = cls

    def register_many(self, model_type: str, names: list[str], cls: Type) -> None:
        for name in names:
            self.register(model_type, name, cls)

    def get(self, model_type: str, name: str) -> Type:
        try:
            return self._models[model_type][self._key(name)]
        except KeyError as exc:
            available = sorted(self._models.get(model_type, {}))
            raise KeyError(
                f"Unknown {model_type} model '{name}'. Available: {available}"
            ) from exc


registry = Registry()


def register_defaults() -> None:
    """Register built-in models lazily to avoid import cycles."""
    if registry._models:
        return

    from energy_fault_detector.models.multilayer_autoencoder import MultilayerAutoencoder
    from energy_fault_detector.models.conditional_autoencoder import ConditionalAE
    from energy_fault_detector.models.vrae import VRAE
    from energy_fault_detector.evaluation.rmse_score import RMSEScore
    from energy_fault_detector.evaluation.mahalanobis_score import MahalanobisScore
    from energy_fault_detector.threshold_selectors.quantile_threshold import QuantileThresholdSelector
    from energy_fault_detector.threshold_selectors.fbeta_threshold import FbetaSelector
    from energy_fault_detector.threshold_selectors.fdr_threshold import FDRSelector
    from energy_fault_detector.threshold_selectors.adaptive_threshold import AdaptiveThresholdSelector

    registry.register_many(
        "autoencoder",
        ["default", "multilayer", "multilayer_autoencoder", "MultilayerAutoencoder"],
        MultilayerAutoencoder,
    )
    registry.register_many(
        "autoencoder",
        ["conditional", "conditional_autoencoder", "ConditionalAE"],
        ConditionalAE,
    )
    registry.register_many("autoencoder", ["vrae", "VRAE"], VRAE)
    registry.register_many("anomaly_score", ["rmse", "RMSEScore"], RMSEScore)
    registry.register_many("anomaly_score", ["mahalanobis", "MahalanobisScore"], MahalanobisScore)
    registry.register_many(
        "threshold_selector",
        ["quantile", "quantile_threshold", "QuantileThresholdSelector"],
        QuantileThresholdSelector,
    )
    registry.register_many("threshold_selector", ["fbeta", "FbetaSelector"], FbetaSelector)
    registry.register_many("threshold_selector", ["fdr", "FDRSelector"], FDRSelector)
    registry.register_many(
        "threshold_selector",
        ["adaptive", "adaptive_threshold", "AdaptiveThresholdSelector"],
        AdaptiveThresholdSelector,
    )
