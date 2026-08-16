"""Variational Recurrent Autoencoder (VRAE)."""

from typing import List
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
import keras.ops as kops
import keras.random as krandom

from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed

from energy_fault_detector.core.autoencoder import Autoencoder


class _SamplingWithKL(keras.layers.Layer):
    """Reparameterization sampling + KL divergence loss as a proper Keras 3 layer.

    Using keras.ops / keras.random (instead of tf.*) so Keras 3 can trace
    this layer with KerasTensors during functional model construction.
    """

    def __init__(self, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = krandom.normal(shape=kops.shape(z_mean))
        z = z_mean + kops.exp(0.5 * z_log_var) * epsilon
        kl_loss = -0.5 * kops.mean(
            1 + z_log_var - kops.square(z_mean) - kops.exp(z_log_var)
        )
        self.add_loss(self.kl_weight * kl_loss)
        return z

    def get_config(self):
        return {**super().get_config(), 'kl_weight': self.kl_weight}


class VRAE(Autoencoder):
    """Variational Recurrent Autoencoder.

    Uses LSTM encoder/decoder with a variational latent space (μ, σ).
    Accepts standard 2D input (N, features) from the pipeline and internally
    converts to/from 3D sequences (windows, sequence_length, features).

    Args:
        hidden_size: Number of LSTM units in encoder and decoder.
        latent_length: Dimensionality of the latent space.
        dropout_rate: Dropout rate applied inside LSTM layers.
        kl_weight: Scalar weight for the KL divergence term in the loss.
        sequence_length: Sliding-window length used to create 3D sequences.
    """

    def __init__(self,
                 hidden_size: int = 90,
                 hidden_layer_depth: int = 1,
                 latent_length: int = 20,
                 batch_size: int = 64,
                 learning_rate: float = 0.0005,
                 epochs: int = 200,
                 dropout_rate: float = 0.2,
                 loss_name: str = 'mean_squared_error',
                 kl_weight: float = 1.0,
                 sequence_length: int = 200,
                 metrics: List[str] = None,
                 decay_rate: float = None,
                 decay_steps: float = None,
                 early_stopping: bool = False,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 noise: float = 0.0,
                 **kwargs):
        metrics = ['mean_absolute_error'] if metrics is None else metrics
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            loss_name=loss_name,
            metrics=metrics,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            noise=noise,
            **kwargs
        )
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight
        self.sequence_length = sequence_length

    def create_model(self, input_dimension: int, **kwargs) -> KerasModel:
        """Build VRAE: LSTM encoder → (μ, σ) → z (+ KL loss) → LSTM decoder."""
        inputs = Input(shape=(self.sequence_length, input_dimension), name='input')

        encoded = LSTM(
            units=self.hidden_size,
            return_sequences=False,
            dropout=self.dropout_rate,
            name='encoder_lstm',
        )(inputs)

        z_mean = Dense(self.latent_length, name='z_mean')(encoded)
        z_log_var = Dense(self.latent_length, name='z_log_var')(encoded)

        # _SamplingWithKL uses keras.ops so Keras 3 can trace with KerasTensors
        z = _SamplingWithKL(kl_weight=self.kl_weight, name='z')([z_mean, z_log_var])

        repeated_z = RepeatVector(self.sequence_length, name='repeat_z')(z)
        decoded = LSTM(
            units=self.hidden_size,
            return_sequences=True,
            dropout=self.dropout_rate,
            name='decoder_lstm',
        )(repeated_z)
        outputs = TimeDistributed(
            Dense(input_dimension, activation='linear'),
            name='decoder_output',
        )(decoded)

        self.model = KerasModel(inputs=inputs, outputs=outputs, name='VRAE')

        self.encoder = KerasModel(
            inputs=inputs,
            outputs=[z_mean, z_log_var, z],
            name='VRAE_Encoder',
        )

        return self.model

    def compile_model(self, new_learning_rate: float = None, **kwargs):
        """Compile with reconstruction loss only; KL term lives in the sampling layer."""
        optimizer = keras.optimizers.Adam(
            learning_rate=new_learning_rate or self.learning_rate
        )
        self.model.compile(optimizer=optimizer, loss=self.loss_name, metrics=self.metrics)

    # ------------------------------------------------------------------
    # 2D ↔ 3D helpers
    # ------------------------------------------------------------------

    def _to_sequences(self, x: np.ndarray) -> np.ndarray:
        """Sliding-window view: (N, F) → (N - seq_len + 1, seq_len, F)."""
        n = len(x)
        if n < self.sequence_length:
            raise ValueError(
                f"Data has {n} samples but sequence_length={self.sequence_length}. "
                "Reduce sequence_length or provide more data."
            )
        return np.stack([x[i:i + self.sequence_length] for i in range(n - self.sequence_length + 1)])

    def _from_sequences(self, seq_out: np.ndarray, n_original: int) -> np.ndarray:
        """Invert sliding-window: (n_windows, seq_len, F) → (n_original, F).

        First seq_len-1 timesteps come from the first window (positions 0..seq_len-2).
        Remaining timesteps take the last position of each successive window.
        Total rows = (seq_len-1) + n_windows = n_original.
        """
        result = np.empty((n_original, seq_out.shape[2]), dtype=seq_out.dtype)
        prefix = min(self.sequence_length - 1, n_original)
        result[:prefix] = seq_out[0, :prefix]
        for t in range(self.sequence_length - 1, n_original):
            result[t] = seq_out[t - self.sequence_length + 1, -1]
        return result

    # ------------------------------------------------------------------
    # Overrides to handle 2D pipeline input
    # ------------------------------------------------------------------

    def __call__(self, x, conditions=None):
        """TF-differentiable 2D→3D→2D pass for ARCANA GradientTape compatibility.

        x is a real tf.Tensor or numpy array here (not a KerasTensor), so tf.*
        ops are safe to use.
        """
        x = tf.cast(x, tf.float32)
        if len(x.shape) == 3:
            return self.model(x)

        # (N, F) → (F, N) → frame → (F, n_windows, seq_len) → (n_windows, seq_len, F)
        x_T = tf.transpose(x)
        framed = tf.signal.frame(x_T, frame_length=self.sequence_length, frame_step=1)
        sequences = tf.transpose(framed, perm=[1, 2, 0])

        seq_out = self.model(sequences)  # (n_windows, seq_len, F)

        # Rebuild (N, F): prefix from first window + last timestep of each window
        # (seq_len-1) + n_windows = (seq_len-1) + (N - seq_len + 1) = N
        prefix = seq_out[0, :self.sequence_length - 1, :]  # (seq_len-1, F)
        last_steps = seq_out[:, -1, :]                     # (n_windows,  F)
        return tf.concat([prefix, last_steps], axis=0)     # (N, F)

    def _make_dataset(self, x: np.ndarray, stride: int) -> tf.data.Dataset:
        """Build a lazy tf.data.Dataset — only one batch in memory at a time.

        stride=sequence_length → non-overlapping chunks (for training).
        stride=1              → sliding window, one window per timestep (for predict).
        """
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=x,
            targets=None,
            sequence_length=self.sequence_length,
            sequence_stride=stride,
            batch_size=self.batch_size,
        )
        return ds

    def _fit_model(self, x, x_val, epochs, callbacks, **kwargs):
        """Train on non-overlapping sequences (stride=seq_len): same coverage, minimal RAM.

        N samples → N // seq_len sequences instead of N sliding windows.
        Memory: batch_size × seq_len × F  instead of  N × seq_len × F.
        """
        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        # Non-overlapping: divides data into independent chunks, no redundancy
        train_ds = self._make_dataset(x_np, stride=self.sequence_length).map(
            lambda seq: (seq, seq)
        )

        val_ds = None
        if x_val is not None:
            x_val_np = x_val.values if isinstance(x_val, pd.DataFrame) else np.asarray(x_val)
            val_ds = self._make_dataset(x_val_np, stride=self.sequence_length).map(
                lambda seq: (seq, seq)
            )

        fit_history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            **kwargs
        )
        self._extend_fit_history(fit_history.history)

    def _predict(self, x, **kwargs):
        """Predict with stride=1: every timestep gets scored via the window ending at it.

        Processes one batch at a time (no full materialisation). Only the last
        timestep of each window is kept, so output memory stays small.
        """
        verbose = kwargs.get('verbose', 0)
        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        n, n_features = x_np.shape

        if n < self.sequence_length:
            raise ValueError(
                f"Data has {n} samples but sequence_length={self.sequence_length}."
            )

        result = np.empty((n, n_features), dtype=np.float32)

        # First seq_len-1 timesteps: take from first-window prediction
        first_win = x_np[:self.sequence_length][np.newaxis].astype(np.float32)
        first_out = self.model.predict(first_win, verbose=0)  # (1, seq_len, F)
        prefix = min(self.sequence_length - 1, n)
        result[:prefix] = first_out[0, :prefix]

        # Remaining timesteps: last-step output of each sliding window, batch by batch
        n_windows = n - self.sequence_length + 1
        n_batches = int(np.ceil(n_windows / self.batch_size))
        if verbose:
            print(f"VRAE predict: {n} samples → {n_windows} windows, {n_batches} batches")
        ds = self._make_dataset(x_np, stride=1)
        t = self.sequence_length - 1
        for i, batch in enumerate(ds):
            batch_out = self.model(batch, training=False).numpy()  # (B, seq_len, F)
            last_steps = batch_out[:, -1, :]                       # (B, F)
            end = t + len(last_steps)
            result[t:end] = last_steps
            t = end
            if verbose and (i + 1) % 50 == 0:
                print(f"\r  batch {i + 1}/{n_batches}", end='', flush=True)
        if verbose:
            print(f"\r  batch {n_batches}/{n_batches} - done")

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(result, index=x.index, columns=x.columns)
        return result
