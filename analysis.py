from matplotlib.ticker import ScalarFormatter
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.colors as mcolors
from scipy.signal import spectrogram


class Analysis:
    def __init__(
        self, pipeline: AudioPipeline, nfft=1024, window=50, figsize=[16, 2], cmap="jet"
    ):
        self._pipeline = pipeline

        self.__sample_rate = pipeline.sample_rate
        self.__nyquist = pipeline.nyquist

        self.__sample_length = pipeline.sample_length
        self.__sample_length_ms = pipeline.sample_length_ms
        self.__sample_width = pipeline.sample_width
        self.__bit_depth = pipeline.bit_depth
        self.__sample_channels = pipeline.sample_channels

        self.__sample_overlap = pipeline.sample_overlap

        self.nfft = nfft
        self.window = window

        self.figsize = figsize
        self.cmap = cmap

        self._xticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

    def print_dictionary_details(self, dictionary):
        keys = dictionary.keys()
        total_elements = 0

        print(f"Dictionary details - there are {len(keys)} keys.")

        for key in keys:
            nr_elements = len(dictionary[key])
            print(f"> {key}: {nr_elements} elements")
            total_elements += nr_elements

        print(f"...for a total of {total_elements}.")

    def plot_waveform(self, waveform):
        """
        Plots a waveform from a 1D NumPy array.

        Parameters:
        waveform (ndarray): The 1D array representing the sound wave.
        """
        time_axis = np.linspace(
            0, len(waveform) / self.__sample_rate, num=len(waveform)
        )

        plt.figure(figsize=self.figsize)
        plt.plot(time_axis, waveform, linewidth=1)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.title("Waveform")
        plt.grid(True)
        plt.show()

    def plot_waveforms(self, waveforms: list[np.array]):
        """
        Plots multiple sound waveforms in a single matplotlib figure, arranging them in a grid.

        Parameters:
        *waveforms: Variable number of NumPy arrays representing waveforms.
        """
        num_waves = len(waveforms)
        cols = int(np.ceil(np.sqrt(num_waves)))
        rows = int(np.ceil(num_waves / cols))

        fig, axs = plt.subplots(rows, cols, figsize=self.figsize)
        axs = np.array(axs).reshape(-1)

        for i, waveform in enumerate(waveforms):
            waveform = np.array(waveform)
            time = np.linspace(0, len(waveform) / self.__sample_rate, num=len(waveform))
            axs[i].plot(time, waveform)
            axs[i].set_title(f"Waveform {i + 1}")
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        for j in range(i + 1, len(axs)):
            axs[j].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_average_spectrum(self, audio_segments):
        """
        Mixes multiple AudioSegments, computes their combined frequency spectrum,
        and plots it with brightness indicating frequency dominance.

        Args:
            audio_segments (list): List of pydub.AudioSegment objects.
            sample_rate (int): Sample rate for analysis (default: 44100 Hz).
            n_fft (int): FFT window size (default: 2048).
            cmap (str): Matplotlib colormap (default: 'viridis').
        """
        mixed = audio_segments[0]
        for seg in audio_segments[1:]:
            mixed = mixed.overlay(seg)

        samples = np.array(mixed.get_array_of_samples())

        if mixed.channels == 2:
            samples = samples[::2]

        samples = self._pipeline.nparray_to_tanh_normal(samples)

        yf = fft(samples)
        xf = fftfreq(len(samples), 1 / self.__sample_rate)

        half_n = len(xf) // 2
        xf = xf[:half_n]
        yf = np.abs(yf[:half_n])

        plt.figure(figsize=self.figsize)
        plt.specgram(
            samples,
            Fs=self.__sample_rate,
            NFFT=self.nfft,
            cmap=self.cmap,
            mode="magnitude",
        )
        plt.colorbar(label="Intensity (dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Combined Frequency Spectrum")
        plt.show()

    def plot_average_volume(self, audio_segments):
        """
        Plots the average volume (RMS) over time for a list of AudioSegments.

        Args:
            audio_segments (list): List of pydub.AudioSegment objects.
            window_ms (int): Time window (in ms) for volume averaging (default: 50ms).
        """
        if not audio_segments:
            raise ValueError("Input list 'audio_segments' is empty!")

        mixed = audio_segments[0]
        for seg in audio_segments[1:]:
            if seg.frame_count() == 0:
                print("Warning: Skipping empty AudioSegment!")
                continue
            mixed = mixed.overlay(seg)

        samples = np.array(mixed.get_array_of_samples())

        if len(samples) == 0:
            raise ValueError("Mixed audio has zero samples!")

        if mixed.channels == 2:
            samples = samples.reshape(-1, 2)
            samples = samples.mean(axis=1)

        samples = self._pipeline.nparray_to_tanh_normal(samples)

        window_size = int(self.window * self.__sample_rate / 1000)
        rms = []

        for i in range(0, len(samples), window_size):
            chunk = samples[i : i + window_size]
            if len(chunk) == 0:
                continue
            rms.append(np.sqrt(np.mean(chunk**2)))

        times = np.arange(len(rms)) * (self.window / 1000)

        plt.figure(figsize=self.figsize)
        plt.plot(times, rms, color="blue", label="RMS Volume")
        plt.xlabel("Time (s)")
        plt.ylabel("Volume (RMS)")
        plt.title("Average Volume Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_average_magnitude_spectrum(self, audio_segments):
        """
        Mixes multiple AudioSegments, computes their combined frequency spectrum,
        and plots it on a log-frequency axis with human-readable tick labels.
        """
        mixed = audio_segments[0]
        for seg in audio_segments[1:]:
            mixed = mixed.overlay(seg)

        samples = np.array(mixed.get_array_of_samples())

        if mixed.channels == 2:
            samples = samples[::2]

        samples = self._pipeline.nparray_to_tanh_normal(samples)

        yf = fft(samples)
        xf = fftfreq(len(samples), 1 / self.__sample_rate)

        half_n = len(xf) // 2
        xf = xf[:half_n]
        yf = np.abs(yf[:half_n])

        mask = xf > 0
        xf = xf[mask]
        yf = yf[mask]

        plt.figure(figsize=self.figsize)
        plt.plot(xf, yf, color="blue", alpha=0.7)
        plt.xscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title("Combined Magnitude Spectrum (Log Frequency Scale)")

        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().xaxis.set_minor_formatter(ScalarFormatter())
        plt.xticks([tick for tick in self._xticks if tick <= self.__nyquist])

        plt.grid(which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def plot_average_spectrum_log(self, audio_segments):
        mixed = audio_segments[0]
        for seg in audio_segments[1:]:
            mixed = mixed.overlay(seg)

        samples = np.array(mixed.get_array_of_samples())
        if mixed.channels == 2:
            samples = samples[::2]
        samples = self._pipeline.nparray_to_tanh_normal(samples)

        f, t, Sxx = spectrogram(
            samples,
            fs=self.__sample_rate,
            nperseg=self.nfft,
            scaling="spectrum",
            mode="magnitude",
        )

        Sxx_dB = 10 * np.log10(Sxx + 1e-10)

        plt.figure(figsize=self.figsize)
        plt.imshow(
            Sxx_dB,
            aspect="auto",
            origin="lower",
            extent=[t.min(), t.max(), f.min(), f.max()],
            cmap=self.cmap,
        )
        plt.yscale("log")
        plt.colorbar(label="Intensity (dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Combined Frequency Spectrum (Log Scale)")

        log_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        plt.yticks(log_freqs, [str(f) for f in log_freqs])
        plt.grid(True, which="both", linestyle="--", alpha=0.3)

        plt.show()

    def plot_history(self, history):
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
