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
        self.sample_width = 1
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

        for dir_path, label in zip(dir_paths, labels):
            waveforms, file_paths = self.from_directory(dir_path=dir_path)
            waveforms_dict[label] = dict(zip(file_paths, waveforms))

        return waveforms_dict

    def _get_silent_audio_segment(self, length=0):
        if length <= 0:
            length = self.sample_length_ms
        return AudioSegment.silent(duration=length)

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

    def process_waveform(
        self, waveform: AudioSegment, shouldSlice=True, shouldConvert=True
    ):
        """
        Remove DC offset, normalize to 0dB, pad/trim/slice to fixed length.

        Parameters:
        waveform (AudioSegment): Waveform to apply processing to.
        """
        processed = (
            waveform.set_channels(self.sample_channels)
            .set_frame_rate(self.sample_rate)
            .set_sample_width(self.sample_width)
        )

        # sound is too short to slice, pad it
        if len(processed) <= self.sample_length_ms:
            processed = self._get_silent_audio_segment().overlay(processed)

        # sound is longer than one section and we wanna slice it
        elif shouldSlice:
            nr_slices = math.floor(
                len(processed) / (self.sample_length_ms - self.sample_overlap)
            )
            sliced = []
            for i in range(nr_slices):
                # take the next slice of audio
                slice_start_pos = (self.sample_length_ms - self.sample_overlap) * i
                slice_end_pos = slice_start_pos + self.sample_length_ms
                slice = processed[slice_start_pos:slice_end_pos]

                # fade slice in and out, remove dc, normalize
                slice = slice.fade_in(1).fade_out(1)
                slice = slice.high_pass_filter(20).normalize(0)

                # convert to nparray
                if shouldConvert:
                    slice = self.audio_segment_to_nparray(slice)
                    slice = self.nparray_to_tanh_normal(slice)

                # add slice to array of slices
                sliced.append(slice)

            processed = sliced

        # sound is longer than one section, but we just wanna trim
        else:
            processed = processed[: self.sample_length_ms]

        # if array was not sliced, apply final touches individually
        if not shouldSlice:
            processed = processed.fade_in(1).fade_out(1)
            processed = processed.high_pass_filter(20).normalize(0)

            if shouldConvert:
                processed = self.audio_segment_to_nparray(processed)
                processed = self.nparray_to_tanh_normal(processed)

        return processed

    def process_waveform_dictionary(
        self, dict: dict, shouldSlice=True, shouldConvert=True
    ):
        processed = {}
        for key, value in dict.items():
            processed[key] = self.process_waveform(
                value, shouldSlice=shouldSlice, shouldConvert=shouldConvert
            )
        return processed

    def create_dataset_from_dictionaries(self, x_dict: dict, y_dict: list[dict]):
        """
        Supply dictionaries of waveforms to create datasets.
        """
        x = []
        y = []
        for dict in y_dict:
            y.append([])

        for key in x_dict:
            x.append(x_dict[key])
            for i, _ in enumerate(y):
                y[i].append(y_dict[i][key])
        return x, y

    def _get_longest_waveform_length(self, waveforms: list[AudioSegment]):
        longest = 0
        for waveform in waveforms:
            if len(waveform) > longest:
                longest = len(waveform)
        return longest

    def compile_vertical(self, waveforms: list[AudioSegment]):
        vertical_length = self._get_longest_waveform_length(waveforms=waveforms)
        vertical = self._get_silent_audio_segment(vertical_length)

        for waveform in waveforms:
            vertical = vertical.overlay(waveform)
        vertical = vertical.normalize(0)

        return vertical

    def layer_waveform_dictionaries(self, dicts: list[dict]):
        layered = {}

        for i, key in enumerate(dicts[0]):
            waveforms = []

            for dict in dicts:
                list_of_values = list(dict.values())
                waveforms.append(list_of_values[i])

            layered[key] = self.compile_vertical(waveforms=waveforms)

        return layered

    def nparray_to_audio_segment(self, nparray: np.array):
        """
        Convert np.array to AudioSegment.

        Parameters:
        nparray (np.array): Numpy array to be converted.
        """
        int_samples = np.clip(nparray, -1, 1)
        int_samples = (int_samples * 32767).astype(np.int16)
        int_samples = int_samples.tobytes()
        return AudioSegment(
            data=int_samples,
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=self.sample_channels,
        ).normalize(0)

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

    def denormalize_tanh_normal(self, nparray: np.array):
        """
        Denormalize nparray from -1, 1 values to original range.

        Parameters:
        nparray (np.array): Numpy array to be denormalized.
        """
        return (nparray + 1) * self.bit_depth / 2

    def _process_predicted(self, predicted):
        """
        Process predicted samples to AudioSegment.

        Parameters:
        predicted (list): List of predicted samples.
        """
        # processed = self.denormalize_tanh_normal(predicted)
        processed = self.nparrays_to_audio_segments(predicted)
        return processed
