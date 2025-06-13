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

    raw, _ = audio_pipeline.from_directory(
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\1"
    )
    raw_sliced = audio_pipeline.slice_waveform_list(raw)
    processed = audio_pipeline.process_waveform_list(raw_sliced)
    x, _ = audio_pipeline.create_dataset_from_lists(x_list=processed, y_list=processed)

    autoencoder = Autoencoder(pipeline=audio_pipeline, lowest_freq=200)

    autoencoder.generate_autoencoder(latent_dim=8, filters=8, dropout=0.3)
    autoencoder.print_summary()

    def schedule(epoch, lr):
        if epoch < 1:
            return lr
        else:
            return lr * 0.95

    scheduler = LearningRateScheduler(schedule, verbose=1)
    stopper = EarlyStopping(
        monitor="val_loss", patience=1, verbose=1, restore_best_weights=True
    )

    history = autoencoder.train_autoencoder(
        x=x[: int(len(x) * 0.9)],
        y=x[: int(len(x) * 0.9)],
        batch_size=8,
        epochs=3,
        validation_split=0.1,
        learning_rate=0.0005,
        loss=combined_loss,
        callbacks=[scheduler, stopper],
    )

    autoencoder.save_model(
        "C:\\Users\\cools\\Desktop\\datasets\\models\\pretrain_model"
    )

    analysis = Analysis(pipeline=audio_pipeline, nfft=256, window=1, figsize=[8, 6])
    analysis.plot_history(history=history)

    original = processed[int(len(x) * 0.9) :]
    predicted = autoencoder.predict(x[int(len(x) * 0.9) :])

    audio_pipeline.export_audio_segments_to_wav(
        predicted,
        "C:\\Users\\cools\\Desktop\\datasets\\models\\pretrain_model\\predicted",
    )
    audio_pipeline.export_audio_segments_to_wav(
        original,
        "C:\\Users\\cools\\Desktop\\datasets\\models\\pretrain_model\\original",
    )

    analysis.plot_average_volume(original)
    analysis.plot_average_spectrum_log(original)
    analysis.plot_average_magnitude_spectrum(original)

    analysis.plot_average_volume(predicted)
    analysis.plot_average_spectrum_log(predicted)
    analysis.plot_average_magnitude_spectrum(predicted)
