"""Concrete fault detector pipeline built from configured components."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from energy_fault_detector.config import Config
from energy_fault_detector.core.fault_detection_model import FaultDetectionModel
from energy_fault_detector.core.fault_detection_result import FaultDetectionResult, ModelMetadata
from energy_fault_detector.root_cause_analysis.arcana import Arcana


PathLike = Union[str, Path]


class FaultDetector(FaultDetectionModel):
    """Autoencoder-based anomaly detector used by the wind-turbine experiments."""

    def _normal_index(self, sensor_data: pd.DataFrame, normal_index: Optional[pd.Series]) -> pd.Series:
        if normal_index is None:
            return pd.Series(True, index=sensor_data.index)
        return normal_index.reindex(sensor_data.index).fillna(False).astype(bool)

    def preprocess_train_data(
        self,
        sensor_data: pd.DataFrame,
        normal_index: Optional[pd.Series] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.Series]:
        """Fit preprocessing on train data and return transformed train/validation data."""
        normal_index = self._normal_index(sensor_data, normal_index)
        train_raw, val_raw = self.train_val_split(sensor_data)
        train_normal = normal_index.reindex(train_raw.index).fillna(False).astype(bool)

        fit_raw = train_raw.loc[train_normal]
        if fit_raw.empty:
            fit_raw = train_raw

        self.data_preprocessor.fit(fit_raw)
        train_data = self.data_preprocessor.transform(train_raw)
        val_data = self.data_preprocessor.transform(val_raw) if val_raw is not None and len(val_raw) else None
        return train_data, val_data, train_normal

    def fit(
        self,
        sensor_data: pd.DataFrame,
        normal_index: pd.Series = None,
        save_models: bool = True,
        overwrite_models: bool = False,
        fit_autoencoder_only: bool = False,
        save_model: Optional[bool] = None,
        **kwargs,
    ) -> ModelMetadata:
        """Fit preprocessing, autoencoder, anomaly score, and threshold selector."""
        if save_model is not None:
            save_models = save_model

        train_data, val_data, train_normal = self.preprocess_train_data(sensor_data, normal_index)
        autoencoder_train_data = train_data.loc[train_normal]
        if autoencoder_train_data.empty:
            autoencoder_train_data = train_data

        self.autoencoder.fit(autoencoder_train_data, x_val=val_data, verbose=self.config.verbose)

        train_recon_error = self.autoencoder.get_reconstruction_error(train_data, verbose=0)
        self.anomaly_score.fit(train_recon_error, y=train_normal)
        train_scores = self.anomaly_score.transform(train_recon_error)

        val_recon_error = None
        if val_data is not None:
            val_recon_error = self.autoencoder.get_reconstruction_error(val_data, verbose=0)

        if not fit_autoencoder_only:
            if self.config.fit_threshold_on_val and val_recon_error is not None:
                val_normal = self._normal_index(sensor_data, normal_index).reindex(val_data.index).fillna(False)
                val_scores = self.anomaly_score.transform(val_recon_error)
                self.threshold_selector.fit(val_scores, y=val_normal)
            else:
                self.threshold_selector.fit(train_scores, y=train_normal)

        model_path = ""
        model_date = ""
        if save_models:
            model_path, model_date = self.save_models(overwrite=overwrite_models)

        return ModelMetadata(
            model_date=model_date,
            model_path=model_path,
            train_recon_error=train_recon_error,
            val_recon_error=val_recon_error,
        )

    def predict(
        self,
        sensor_data: pd.DataFrame,
        model_path: Optional[str] = None,
        asset_id: Union[int, str] = None,
        root_cause_analysis: Optional[bool] = None,
        track_losses: bool = False,
        track_bias: bool = False,
    ) -> FaultDetectionResult:
        """Predict anomalies for sensor data using fitted or loaded models."""
        if model_path is not None:
            self.load_models(model_path)

        transformed = self.data_preprocessor.transform(sensor_data)
        reconstruction = self.autoencoder.predict(transformed, verbose=0)
        recon_error = self.autoencoder.get_reconstruction_error(transformed, verbose=0)
        anomaly_score = self.anomaly_score.transform(recon_error)
        predicted = self.threshold_selector.predict(anomaly_score)
        predicted = pd.Series(predicted, index=transformed.index, name="anomaly").astype(bool)

        try:
            reconstruction_out = self.data_preprocessor.inverse_transform(reconstruction)
        except Exception:
            reconstruction_out = reconstruction

        bias_data = arcana_losses = tracked_bias = None
        should_run_arcana = self.config.root_cause_analysis if root_cause_analysis is None else root_cause_analysis
        if should_run_arcana:
            bias_data, arcana_losses, tracked_bias = self.run_root_cause_analysis(
                sensor_data=sensor_data,
                track_losses=track_losses,
                track_bias=track_bias,
            )

        return FaultDetectionResult(
            predicted_anomalies=predicted,
            reconstruction=reconstruction_out,
            recon_error=recon_error,
            anomaly_score=anomaly_score,
            bias_data=bias_data,
            arcana_losses=arcana_losses,
            tracked_bias=tracked_bias,
        )

    def run_root_cause_analysis(
        self,
        sensor_data: pd.DataFrame,
        track_losses: bool = False,
        track_bias: bool = False,
    ):
        transformed = self.data_preprocessor.transform(sensor_data)
        arcana = Arcana(self.autoencoder, **self.config.arcana_params)
        return arcana.find_arcana_bias(
            transformed,
            track_losses=track_losses,
            track_bias=track_bias,
        )
