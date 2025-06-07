from analysis import Analysis
from audio_pipeline import AudioPipeline
from autoencoder import Autoencoder
from keras._tf_keras.keras.callbacks import LearningRateScheduler, EarlyStopping


if __name__ == "__main__":
    audio_pipeline = AudioPipeline(
        sample_rate=16000,
        sample_length=0.2,
        sample_width=2,
        sample_channels=1,
        sample_overlap=0.0,
    )

    raw, _ = audio_pipeline.from_directories(
        [
            "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\fund",
            "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\trans",
            "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body",
        ],
        ["fund", "trans", "body"],
    )

    raw_layered = audio_pipeline.layer_waveform_lists(
        [raw["fund"], raw["trans"], raw["body"]]
    )

    processed_body = audio_pipeline.process_waveform_list(raw["body"])
    processed_layered = audio_pipeline.process_waveform_list(raw_layered)

    x, y = audio_pipeline.create_dataset_from_lists(
        x_list=processed_layered, y_list=processed_body
    )

    autoencoder = Autoencoder(pipeline=audio_pipeline, lowest_freq=200)

    autoencoder.generate_autoencoder(latent_dim=8, filters=8, dropout=0.3)
    autoencoder.load_model("C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\model")
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

    x_train = x[: int(len(x) * 0.9)]
    y_train = y[: int(len(y) * 0.9)]

    history = autoencoder.train_autoencoder(
        x=x_train,
        y=y_train,
        batch_size=16,
        epochs=10,
        validation_split=0.1,
        learning_rate=0.0005,
        loss="mse",
        callbacks=[scheduler, stopper],
    )

    autoencoder.save_model("C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body_model")

    analysis = Analysis(pipeline=audio_pipeline, nfft=256, window=5, figsize=[8, 6])
    analysis.plot_history(history=history)

    original_x = processed_layered[int(len(x) * 0.9) :]
    original_y = processed_body[int(len(y) * 0.9) :]
    predicted_y = autoencoder.predict(x[int(len(x) * 0.9) :])

    audio_pipeline.export_audio_segments_to_wav(
        predicted_y,
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body_model\\predicted",
    )
    audio_pipeline.export_audio_segments_to_wav(
        original_x,
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body_model\\original_x",
    )
    audio_pipeline.export_audio_segments_to_wav(
        original_y,
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body_model\\original_y",
    )

    analysis.plot_average_volume(original_x)
    analysis.plot_average_spectrum_log(original_x)
    analysis.plot_average_magnitude_spectrum(original_x)

    analysis.plot_average_volume(original_y)
    analysis.plot_average_spectrum_log(original_y)
    analysis.plot_average_magnitude_spectrum(original_y)

    analysis.plot_average_volume(predicted_y)
    analysis.plot_average_spectrum_log(predicted_y)
    analysis.plot_average_magnitude_spectrum(predicted_y)
