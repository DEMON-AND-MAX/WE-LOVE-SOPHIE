import math
import os
import numpy as np
from pydub import AudioSegment


class AudioPipeline:
    def __init__(
        self,
        sample_rate=44100,
        sample_length=1.2,
        sample_width=2,
        sample_channels=1,
        sample_overlap=0.2,
    ):
        self.sample_rate = sample_rate
        self.nyquist = math.floor(sample_rate / 2)

        self.sample_length = sample_length
        self.sample_length_ms = int(sample_length * 1000)
        self.sample_width = 2
        self.bit_depth = 2 ** (8 * sample_width - 1)
        self.sample_channels = sample_channels

        self.sample_overlap = sample_overlap

    def from_file(self, file_path: str):
        """
        Load AudioSegment from file, only accepts .wav.

        Parameters:
        file_path (string): Path to the .wav file.
        """
        if not file_path.endswith(".wav"):
            return None

        waveform = AudioSegment.from_file(file_path)
        waveform = waveform.set_channels(self.sample_channels)
        waveform = waveform.set_frame_rate(self.sample_rate)
        waveform = waveform.set_sample_width(self.sample_width)
        return waveform

    def from_directory(self, dir_path: str):
        """
        Load all AudioSegments from directory, only accepts .wav.

        Parameters:
        dir_path (string): Path to the directory.
        """
        waveforms = []
        waveform_paths = []

        for file in os.listdir(dir_path):
            if file.endswith(".wav"):
                file_path = os.path.join(dir_path, file)
                waveforms.append(self.from_file(file_path=file_path))
                waveform_paths.append(file_path)

        return waveforms, waveform_paths

    def from_directories(self, dir_paths: list[str], labels: list[str]):
        """
        Load AudioSegments from all directories, only accepts .wav.

        Parameters:
        dir_path (string): Path to the directory to be loaded.
        label (string): The label to index waveforms on.
        """
        waveforms_dict = {}
        file_paths_dict = {}

        for dir_path, label in zip(dir_paths, labels):
            waveforms, file_paths = self.from_directory(dir_path=dir_path)
            waveforms_dict[label] = waveforms
            file_paths_dict[label] = file_paths

        return waveforms_dict, file_paths_dict

    def _get_silent_audio_segment(self, length=0):
        if length <= 0:
            length = self.sample_length_ms
        silent = AudioSegment.silent(duration=length)
        silent = silent.set_channels(self.sample_channels)
        silent = silent.set_frame_rate(self.sample_rate)
        silent = silent.set_sample_width(self.sample_width)
        return silent

    def audio_segment_to_nparray(self, waveform: AudioSegment):
        """
        AudioSegment to np.array.

        Parameters:
        waveform (AudioSegment): AudioSegment to be converted.
        """
        return np.array(waveform.get_array_of_samples(), dtype=np.float32)

    def nparray_to_tanh_normal(self, nparray: np.array):
        """
        Normalize nparray to -1, 1 values for tanh activation.
        """
        return nparray / self.bit_depth

    def process_waveform(self, waveform: AudioSegment, shouldNormalize=True):
        """
        Remove DC offset, normalize to 0dB, pad/trim/slice to fixed length.

        Parameters:
        waveform (AudioSegment): Waveform to apply processing to.
        shouldNormalize (bool): Whether to normalize the waveform.
        """
        # sound is too short to be useful, so pad it with silence
        if len(waveform) <= self.sample_length_ms:
            waveform = self._get_silent_audio_segment().overlay(waveform)
        else:
            waveform = waveform[: self.sample_length_ms]

        waveform = waveform.fade_in(1).fade_out(1)
        waveform = waveform.high_pass_filter(20)

        if shouldNormalize:
            waveform = waveform.normalize(0)

        return waveform

    def slice_waveform(self, waveform: AudioSegment):
        """
        Slice waveform.

        Parameters:
        waveform (AudioSegment): Waveform to be sliced.
        """
        sliced = []
        nr_slices = math.floor(
            len(waveform) / (self.sample_length_ms - self.sample_overlap * 1000)
        )

        for i in range(nr_slices + 1):
            start_pos = int(i * (self.sample_length_ms - self.sample_overlap))
            end_pos = int(start_pos + self.sample_length_ms)
            sliced.append(waveform[start_pos:end_pos])

        return sliced

    def slice_waveform_list(self, waveforms: list[AudioSegment]):
        """
        Slice list of waveforms.

        Parameters:
        waveforms (list): List of waveforms to be sliced.
        """
        sliced = []
        for waveform in waveforms:
            sliced.extend(self.slice_waveform(waveform=waveform))
        return sliced

    def process_waveform_list(self, waveforms: list[AudioSegment]):
        """
        Process list of waveforms.

        Parameters:
        waveforms (list): List of waveforms to be processed.
        """
        processed = []
        for waveform in waveforms:
            processed.append(self.process_waveform(waveform=waveform))
        return processed

    def layer_waveform_lists(self, waveforms: list, noise: float = 0.0):
        """
        Layer list of waveforms.

        Parameters:
        waveforms (list): List of waveform lists to be layered.
        """
        layered = []
        for i in range(len(waveforms[0])):
            vertical = []
            for list in waveforms:
                vertical.append(list[i])
            if noise > 0:
                base_length = len(vertical[0])
                silent = self._get_silent_audio_segment(length=base_length)
                silent_np = self.audio_segment_to_nparray(silent)

                noise_np = (
                    np.random.normal(0, 1, len(silent_np)) * self.bit_depth * noise
                )
                noise_np = np.clip(noise_np, -self.bit_depth, self.bit_depth).astype(
                    np.float32
                )

                noise_segment = self.nparray_to_audio_segment(noise_np / self.bit_depth)

                vertical.append(noise_segment.apply_gain(-60))
            layered.append(self.compile_vertical(waveforms=vertical))
        return layered

    def create_dataset_from_lists(self, x_list: list, y_list: list):
        """
        Supply dictionaries of waveforms to create numpy datasets.
        """
        x = []
        y = []
        for x_wave, y_wave in zip(x_list, y_list):
            x_wave = self.audio_segment_to_nparray(waveform=x_wave)
            y_wave = self.audio_segment_to_nparray(waveform=y_wave)
            x_wave = self.nparray_to_tanh_normal(nparray=x_wave)
            y_wave = self.nparray_to_tanh_normal(nparray=y_wave)
            x.append(x_wave)
            y.append(y_wave)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        x = x[..., np.newaxis]
        y = y[..., np.newaxis]
        return x, y

    def _get_longest_waveform_length(self, waveforms: list[AudioSegment]):
        longest = 0
        for waveform in waveforms:
            if len(waveform) > longest:
                longest = len(waveform)
        return longest

    def compile_vertical(self, waveforms: list[AudioSegment], normalize=True):
        vertical_length = self._get_longest_waveform_length(waveforms=waveforms)
        vertical = self._get_silent_audio_segment(vertical_length)

        for waveform in waveforms:
            vertical = vertical.overlay(waveform)
        if normalize:
            vertical = vertical.normalize(0)

        return vertical

    def nparray_to_audio_segment(self, nparray: np.array):
        """
        Convert np.array to AudioSegment.

        Parameters:
        nparray (np.array): Numpy array to be converted.
        """
        int_samples = np.clip(nparray, -1, 1)
        int_samples = (int_samples * 32767).astype(np.int16)
        int_samples = int_samples.tobytes()
        waveform = AudioSegment(
            data=int_samples,
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=self.sample_channels,
        )
        waveform = self.process_waveform(waveform=waveform, shouldNormalize=True)
        return waveform

    def nparrays_to_audio_segments(self, nparrays: list[np.array]):
        """
        Convert list of np.arrays to AudioSegments.

        Parameters:
        nparrays (list): List of numpy arrays to be converted.
        """
        audio_segments = []
        for nparray in nparrays:
            audio_segments.append(self.nparray_to_audio_segment(nparray))
        return audio_segments

    def export_audio_segments_to_wav(
        self, waveforms: list[AudioSegment], output_path: str
    ):
        """
        Convert samples to .wav files.

        Parameters:
        samples (list): List of samples to be converted.
        output_path (string): Path to the output directory.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i, waveform in enumerate(waveforms):
            waveform.export(os.path.join(output_path, f"sample_{i}.wav"), format="wav")

    def _process_predicted(self, predicted):
        """
        Process predicted samples to AudioSegment.

        Parameters:
        predicted (list): List of predicted samples.
        """
        processed = self.nparrays_to_audio_segments(predicted)
        return processed
