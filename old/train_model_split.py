import numpy as np

from old.audio_pipeline_old import AudioPipeline

from analysis import Analysis
from autoencoder import Autoencoder


directories = [
    "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\fundamental",
    "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\transient",
    "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body",
    "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\layered",
]
labels = ["fundamental", "transient", "body", "layered"]

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

    layered = audio_pipeline.layer_waveform_dictionaries(
        [raw["fundamental"], raw["transient"], raw["body"]],
        shouldSlice=True,
        shouldConvert=True,
    )

    processed_fund = audio_pipeline.process_waveform_dictionary(
        raw["fundamental"], shouldSlice=False, shouldConvert=True
    )
    processed_fund_segment = audio_pipeline.process_waveform_dictionary(
        raw["fundamental"], shouldSlice=False, shouldConvert=False
    )

    processed_trans = audio_pipeline.process_waveform_dictionary(
        raw["transient"], shouldSlice=False, shouldConvert=True
    )
    processed_trans_segment = audio_pipeline.process_waveform_dictionary(
        raw["transient"], shouldSlice=False, shouldConvert=False
    )

    processed_body = audio_pipeline.process_waveform_dictionary(
        raw["body"], shouldSlice=False, shouldConvert=True
    )
    processed_body_segment = audio_pipeline.process_waveform_dictionary(
        raw["body"], shouldSlice=False, shouldConvert=False
    )

    x, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_layered,
        y_dict=[processed_layered],
    )
    x_segment, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_layered_segment,
        y_dict=[processed_layered_segment],
    )

    y, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_fund,
        y_dict=[processed_fund],
    )
    y_segment, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_fund_segment,
        y_dict=[processed_fund_segment],
    )

    analysis = Analysis(
        pipeline=audio_pipeline, nfft=256, window=2, figsize=[8, 5], cmap="jet"
    )

    analysis.plot_waveforms(x[:6])
    analysis.plot_waveforms(y[:6])
    analysis.plot_average_volume(x_segment[:3000])
    analysis.plot_average_volume(y_segment[:3000])
    analysis.plot_average_spectrum_log(x_segment[:3000])
    analysis.plot_average_spectrum_log(y_segment[:3000])
    analysis.plot_average_magnitude_spectrum(x_segment[:3000])
    analysis.plot_average_magnitude_spectrum(y_segment[:3000])

    autoencoder = Autoencoder(pipeline=audio_pipeline, lowest_freq=80)

    autoencoder.generate_autoencoder(
        latent_dim=8, filters=8, dropout=0.3, leave_skips=0
    )
    autoencoder.print_summary()
    autoencoder.load_model("C:\\Users\\cools\\Desktop\\datasets\\dataset_a\\model")

    # train for fundamental extraction (as test)
    history = autoencoder.train_autoencoder(
        x=np.array(x[:3000]),
        y=np.array(y[:3000]),
        batch_size=16,
        epochs=3,
        validation_split=0.1,
        learning_rate=0.00005,
        loss="mse",
    )

    analysis.plot_history(history)

    autoencoder.save_model(
        save_dir="C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\model_fundamental"
    )

    predicted = autoencoder.predict(np.array(x[3000:]))

    analysis.plot_average_volume(predicted)
    analysis.plot_average_spectrum_log(predicted)
    analysis.plot_average_magnitude_spectrum(predicted)

    audio_pipeline.export_audio_segments_to_wav(
        x_segment[:3000], "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\original"
    )
    audio_pipeline.export_audio_segments_to_wav(
        y_segment[:3000],
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\original_y",
    )
    audio_pipeline.export_audio_segments_to_wav(
        predicted, "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\predicted"
    )
