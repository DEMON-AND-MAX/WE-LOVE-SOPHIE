import math
import os
import numpy as np
import tensorflow as tf
import keras._tf_keras.keras
from keras._tf_keras.keras.layers import (
    Input,
    Conv2D,
    ReLU,
    BatchNormalization,
    Flatten,
    Dense,
    Reshape,
    Conv2DTranspose,
    Activation,
    Dropout,
    LeakyReLU,
    LayerNormalization,
    Conv1DTranspose,
    Conv1D,
    Lambda,
    Concatenate,
    UpSampling1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    LayerNormalization,
    GRU,
    RepeatVector,
    TimeDistributed,
)
from keras._tf_keras.keras.initializers import HeNormal
from keras._tf_keras.keras.activations import sigmoid
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import MeanSquaredError, MeanAbsoluteError
from keras._tf_keras.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    Callback,
)
from keras._tf_keras.keras.regularizers import l2
from keras import backend as K
from keras._tf_keras.keras.models import Model, load_model


class Autoencoder:
    def __init__(self, pipeline=None, lowest_freq=80):
        self._pipeline = pipeline

        self.__sample_rate = pipeline.sample_rate
        self.__nyquist = pipeline.nyquist

        self.__sample_length = pipeline.sample_length
        self.__sample_length_ms = pipeline.sample_length_ms
        self.__sample_width = pipeline.sample_width
        self.__bit_depth = pipeline.bit_depth
        self.__sample_channels = pipeline.sample_channels

        self.__sample_overlap = pipeline.sample_overlap

        self.lowest_freq = lowest_freq

        self.input_shape = self._generate_input_shape()
        self.kernels = self._generate_model_kernel_sizes(lowest_freq=lowest_freq)

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        self.latent_dim = 32
        self.filters = 8
        self.dropout = 0.3
        self.leave_skips = 0

        self.learning_rate = 0.01
        self.loss = "mse"

    def _generate_input_shape(self):
        return (
            math.floor(self.__sample_rate * self.__sample_length),
            self.__sample_channels,
        )

    def _generate_model_kernel_sizes(self, lowest_freq=80):
        current_freq = lowest_freq
        kernels = []
        while current_freq <= self.__nyquist:
            kernel_size = math.floor(self.__sample_rate / current_freq)
            kernels.append(kernel_size)
            current_freq *= 2
        return kernels

    def _generate_encoder(
        self, input_shape, kernels, latent_dim=32, filters=8, dropout=0.3
    ):
        x = Input(shape=input_shape, name="Encoder_Input")

        encoder_input = x
        encoder_skips = []

        for i, kernel in enumerate(kernels):
            x = Conv1D(
                filters,
                kernel,
                padding="same",
                activation="tanh",
                name=f"Encoder_Conv1D_{i}",
            )(x)
            x = Dropout(dropout, name=f"Encoder_Dropout_{i}")(x)
            encoder_skips.append(x)

        x = GlobalAveragePooling1D(name=f"Encoder_GlobalAveragePooling1D")(x)
        x = LayerNormalization(name=f"Encoder_LayerNormalization")(x)
        latent = Dense(latent_dim, activation="tanh", name="Encoder_Latent")(x)

        encoder = Model(encoder_input, [latent] + encoder_skips, name="Encoder")
        return encoder, encoder_skips

    def _generate_decoder(
        self,
        output_shape,
        kernels,
        skips,
        latent_dim=32,
        filters=8,
        dropout=0.3,
        trail=2,
        trail_start=3,
        leave_skips=0,
    ):
        decoder_input = Input(shape=(latent_dim,), name="Decoder_Input")

        x = Dense(filters * output_shape[0], activation="tanh", name="Decoder_Dense")(
            decoder_input
        )
        x = Reshape((output_shape[0], filters), name="Decoder_Reshape")(x)

        decoder_skips = []
        adjusted_filters = filters

        for i, kernel in reversed(list(enumerate(kernels))):
            if i <= trail_start - 1:
                adjusted_filters -= trail
            if filters <= 0:
                adjusted_filters = 1

            x = Conv1DTranspose(
                adjusted_filters,
                kernel,
                padding="same",
                activation="tanh",
                name=f"Decoder_Conv1DTranspose_{i}",
            )(x)
            x = Dropout(dropout, name=f"Decoder_Dropout_{i}")(x)

            if i > leave_skips - 1:
                skip = Input(shape=(output_shape[0], filters))
                decoder_skips.insert(0, skip)
                x = Concatenate(name=f"Decoder_Concatenate_{i}")([skip, x])
                x = Dropout(dropout, name=f"Decoder_Dropout_Skip_{i}")(x)

        decoder_output = Conv1DTranspose(
            1, 3, padding="same", activation="tanh", name="Decoder_Output"
        )(x)
        decoder = Model([decoder_input] + decoder_skips, decoder_output, name="Decoder")
        return decoder

    def generate_autoencoder(
        self, latent_dim=32, filters=8, dropout=0.3, leave_skips=0
    ):
        encoder, skips = self._generate_encoder(
            self.input_shape, self.kernels, latent_dim, filters, dropout
        )
        used_skips = len(skips) - leave_skips
        decoder = self._generate_decoder(
            self.input_shape,
            self.kernels,
            skips[:used_skips],
            latent_dim,
            filters,
            dropout,
        )

        autoencoder_input = Input(self.input_shape, name="Autoencoder_Input")
        latent, *encoder_skips = encoder(autoencoder_input)
        decoder_output = decoder([latent] + encoder_skips[:used_skips])

        autoencoder = Model(autoencoder_input, decoder_output, name="Autoencoder")

        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder

        self.latent_dim = latent_dim
        self.filters = filters
        self.dropout = dropout
        self.leave_skips = leave_skips

        return encoder, decoder, autoencoder

    def print_summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()

    def train_autoencoder(
        self,
        x,
        y,
        batch_size=16,
        epochs=5,
        validation_split=0.1,
        learning_rate=0.01,
        loss="mse",
        callbacks=[],
    ):
        self.autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss)
        self.learning_rate = learning_rate
        self.loss = loss

        history = self.autoencoder.fit(
            x,
            y,
            batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
        )

        return history

    def save_model(self, save_dir=""):
        os.makedirs(save_dir, exist_ok=True)

        encoder_path = os.path.join(save_dir, "encoder")
        decoder_path = os.path.join(save_dir, "decoder")
        autoencoder_path = os.path.join(save_dir, "autoencoder")

        os.makedirs(encoder_path, exist_ok=True)
        os.makedirs(decoder_path, exist_ok=True)
        os.makedirs(autoencoder_path, exist_ok=True)

        self.encoder.save(encoder_path + "/encoder.h5")
        self.decoder.save(decoder_path + "/decoder.h5")
        self.autoencoder.save(autoencoder_path + "/autoencoder.h5")

    def load_model(self, load_path=""):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"The specified path '{load_path}' does not exist.")

        self.encoder = load_model(
            os.path.join(load_path, "encoder", "encoder.h5"), compile=False
        )
        self.decoder = load_model(
            os.path.join(load_path, "decoder", "decoder.h5"), compile=False
        )
        self.autoencoder = load_model(
            os.path.join(load_path, "autoencoder", "autoencoder.h5"), compile=False
        )

        self.encoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss
        )
        self.decoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss
        )
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss
        )

        print(f"Model loaded from '{load_path}'.")

    def predict(self, x):
        if self.autoencoder is None:
            raise ValueError(
                "Autoencoder model is not initialized. Please load or create a model first."
            )

        predicted = self.autoencoder.predict(x)
        predicted = predicted.squeeze(-1)
        predicted = self._pipeline._process_predicted(predicted)

        return predicted

    def morph(self, x1, x2, alpha):
        if self.autoencoder is None:
            raise ValueError("Autoencoder model is not initialized.")

        # 1) run both inputs through the encoder (returns [latent, skip1, skip2, …])
        preds_x1 = self.encoder.predict(x1)
        preds_x2 = self.encoder.predict(x2)

        # 2) pull out just the bottleneck vectors
        latent_x1 = np.asarray(preds_x1[0])
        latent_x2 = np.asarray(preds_x2[0])
        if latent_x1.shape != latent_x2.shape:
            raise ValueError(
                f"Latent shapes must match, got {latent_x1.shape} vs {latent_x2.shape}"
            )

        # 3) element‑wise interpolate the latent codes
        morphed_latent = (1 - alpha) * latent_x1 + alpha * latent_x2

        # 4) leave all skip‑connection outputs untouched (take from x1, or x2 if you prefer)
        skips = preds_x1[1:]  # e.g. [skip1_x1, skip2_x1, ...]

        # 5) build the full input list for the decoder: [morphed_latent, *skips]
        decoder_inputs = [morphed_latent] + skips

        # 6) run it through the (multi‑input) decoder
        morphed = self.decoder.predict(decoder_inputs)

        # 7) post‑process exactly as before
        if morphed.shape[-1] == 1:
            morphed = morphed.squeeze(-1)
        return self._pipeline._process_predicted(morphed)
