import numpy as np
from old.audio_pipeline_old import AudioPipeline
from analysis import Analysis
from autoencoder import Autoencoder


directories = ["D:\\Downloads\\kicks-20250220T120427Z-001\\kicks"]
labels = ["kicks"]

if __name__ == "__main__":
    print("Audio Pipeline")
    print("Loading Audio Pipeline...")
    audio_pipeline = AudioPipeline(
        sample_rate=16000,
        sample_length=1.2,
        sample_width=2,
        sample_channels=1,
        sample_overlap=0.2,
    )

    print("Audio Pipeline Loaded")
    print("Loading Audio Files...")
    raw_audio_dict = audio_pipeline.from_directories(
        dir_paths=directories,
        labels=labels,
    )
    print("Audio Files Loaded")

    # here im missing the layering step
    # layered_audio_dict = audio_pipeline.layer_audio_dictionary(
    #     raw_audio_dict, shouldSlice=True, shouldConvert=True
    # )

    print("Processing Audio Files...")
    processed_audio_dict = audio_pipeline.process_waveform_dictionary(
        raw_audio_dict["kicks"], shouldSlice=False, shouldConvert=True
    )
    processed_audio_dict_segment = audio_pipeline.process_waveform_dictionary(
        raw_audio_dict["kicks"], shouldSlice=False, shouldConvert=False
    )
    print("Audio Files Processed")

    print("Creating Dataset...")
    x, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_audio_dict,
        y_dict=[processed_audio_dict],
    )
    x_segment, _ = audio_pipeline.create_dataset_from_dictionaries(
        x_dict=processed_audio_dict_segment,
        y_dict=[processed_audio_dict_segment],
    )
    print("Dataset Created")

    print("Creating Information...")
    analysis = Analysis(
        pipeline=audio_pipeline, nfft=256, window=100, figsize=[5, 3], cmap="jet"
    )

    analysis.plot_waveforms(x[:1])
    analysis.plot_average_volume(x_segment)
    analysis.plot_average_spectrum_log(x_segment)
    analysis.plot_average_magnitude_spectrum(x_segment)

    print(np.array(x).shape)

    autoencoder = Autoencoder(pipeline=audio_pipeline, lowest_freq=80)

    autoencoder.generate_autoencoder(
        latent_dim=8, filters=8, dropout=0.3, leave_skips=0
    )
    autoencoder.print_summary()
    """history = autoencoder.train_autoencoder(
        x=np.array(x[:1000]),
        y=np.array(x[:1000]),
        batch_size=16,
        epochs=3,
        validation_split=0.1,
        learning_rate=0.0005,
        loss="mse",
    )

    analysis.plot_history(history)"""

    autoencoder.load_model("D:\\Downloads\\kicks-20250220T120427Z-001\\models")

    # autoencoder.save_model(save_dir="D:\\Downloads\\kicks-20250220T120427Z-001\\models")
    print(np.array(x[1000:1010]).shape)

    predicted = autoencoder.predict(np.array(x[1000:1010]))

    analysis.plot_average_volume(predicted)
    analysis.plot_average_spectrum_log(predicted)
    analysis.plot_average_magnitude_spectrum(predicted)

    audio_pipeline.export_audio_segments_to_wav(
        x_segment[1000:1010], "D:\\Downloads\\kicks-20250220T120427Z-001\\original"
    )
    audio_pipeline.export_audio_segments_to_wav(
        predicted, "D:\\Downloads\\kicks-20250220T120427Z-001\\predicted"
    )
