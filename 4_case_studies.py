import numpy as np
import tensorflow as tf
from analysis import Analysis
from audio_pipeline import AudioPipeline
from autoencoder import Autoencoder
from keras._tf_keras.keras.callbacks import LearningRateScheduler, EarlyStopping


def spectral_loss(y_true, y_pred, fft_size=256, hop_size=128):
    def _stft(x):
        return tf.signal.stft(
            x, frame_length=fft_size, frame_step=hop_size, pad_end=True
        )

    y_true_spec = tf.abs(_stft(y_true))
    y_pred_spec = tf.abs(_stft(y_pred))

    y_true_log = tf.math.log1p(y_true_spec)
    y_pred_log = tf.math.log1p(y_pred_spec)

    return tf.reduce_mean(tf.abs(y_true_log - y_pred_log))


def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    spec = spectral_loss(y_true, y_pred)
    return mse + 1 * spec


if __name__ == "__main__":
    audio_pipeline = AudioPipeline(
        sample_rate=16000,
        sample_length=0.2,
        sample_width=2,
        sample_channels=1,
        sample_overlap=0.0,
    )

    analysis = Analysis(pipeline=audio_pipeline, nfft=256, window=1, figsize=[8, 6])

    raw, _ = audio_pipeline.from_directory(
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_c"
    )
    raw_sliced = audio_pipeline.slice_waveform_list(raw)
    processed = audio_pipeline.process_waveform_list(raw_sliced)
    x, _ = audio_pipeline.create_dataset_from_lists(x_list=processed, y_list=processed)

    autoencoder = Autoencoder(pipeline=audio_pipeline, lowest_freq=200)

    autoencoder.generate_autoencoder(
        latent_dim=8, filters=8, dropout=0.3, leave_skips=0
    )
    autoencoder.load_model(
        "C:\\Users\\cools\\Desktop\\datasets\\models\\fine_tuned_model"
    )

    original_x = processed[int(len(x) * 0.9) :]
    predicted_y = autoencoder.predict(x[int(len(x) * 0.9) :])

    audio_pipeline.export_audio_segments_to_wav(
        predicted_y,
        "C:\\Users\\cools\\Desktop\\datasets\\case_studies\\predicted",
    )
    audio_pipeline.export_audio_segments_to_wav(
        original_x,
        "C:\\Users\\cools\\Desktop\\datasets\\case_studies\\original_x",
    )

    analysis.plot_average_volume(original_x)
    analysis.plot_average_spectrum_log(original_x)
    analysis.plot_average_magnitude_spectrum(original_x)

    analysis.plot_average_volume(predicted_y)
    analysis.plot_average_spectrum_log(predicted_y)
    analysis.plot_average_magnitude_spectrum(predicted_y)
