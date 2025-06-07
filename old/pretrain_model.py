import numpy as np

from old.audio_pipeline_old import AudioPipeline

from analysis import Analysis
from autoencoder import Autoencoder


directories = ["C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\1"]
labels = ["dataset"]

if __name__ == "__main__":
    audio_pipeline = AudioPipeline(
        sample_rate=16000,
        sample_length=1.2,
        sample_width=2,
        sample_channels=1,
        sample_overlap=0.2,
    )

    raw = audio_pipeline.from_directories(
        dir_paths=directories,
        labels=labels,
    )

    processed = audio_pipeline.process_waveform_dictionary(
        raw["dataset"], shouldSlice=False, shouldConvert=True
    )
    processed_segment = audio_pipeline.process_waveform_dictionary(
        raw["dataset"], shouldSlice=False, shouldConvert=False
    )

    x, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed,
        y_dict=[processed],
    )
    x_segment, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_segment,
        y_dict=[processed_segment],
    )

    analysis = Analysis(
        pipeline=audio_pipeline, nfft=256, window=2, figsize=[8, 5], cmap="jet"
    )

    analysis.plot_waveforms(x[:6])
    analysis.plot_average_volume(x_segment)
    analysis.plot_average_spectrum_log(x_segment)
    analysis.plot_average_magnitude_spectrum(x_segment)

    autoencoder = Autoencoder(pipeline=audio_pipeline, lowest_freq=80)

    autoencoder.generate_autoencoder(
        latent_dim=8, filters=8, dropout=0.3, leave_skips=0
    )
    autoencoder.print_summary()
    history = autoencoder.train_autoencoder(
        x=np.array(x[:2000]),
        y=np.array(x[:2000]),
        batch_size=16,
        epochs=3,
        validation_split=0.1,
        learning_rate=0.00005,
        loss="mse",
    )

    analysis.plot_history(history)

    autoencoder.save_model(
        save_dir="C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\model"
    )

    predicted = autoencoder.predict(np.array(x[2000:]))

    analysis.plot_average_volume(predicted)
    analysis.plot_average_spectrum_log(predicted)
    analysis.plot_average_magnitude_spectrum(predicted)

    audio_pipeline.export_audio_segments_to_wav(
        x_segment[:2000], "C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\original"
    )
    audio_pipeline.export_audio_segments_to_wav(
        predicted, "C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\predicted"
    )
