import os
from pydub import AudioSegment


def layer_audio_from_folders(folder_a, folder_b, folder_c, output_folder):
    """
    Layers corresponding audio files from three folders and saves the results.

    Parameters:
        folder_a, folder_b, folder_c (str): Paths to input folders.
        output_folder (str): Path to output folder where layered sounds will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Sort file lists to ensure matching order
    files_a = sorted(
        [
            f
            for f in os.listdir(folder_a)
            if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a"))
        ]
    )
    files_b = sorted(
        [
            f
            for f in os.listdir(folder_b)
            if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a"))
        ]
    )
    files_c = sorted(
        [
            f
            for f in os.listdir(folder_c)
            if f.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a"))
        ]
    )

    if not (len(files_a) == len(files_b) == len(files_c)):
        raise ValueError(
            "All three folders must contain the same number of audio files."
        )

    for i in range(len(files_a)):
        try:
            a = AudioSegment.from_file(os.path.join(folder_a, files_a[i]))
            b = AudioSegment.from_file(os.path.join(folder_b, files_b[i]))
            c = AudioSegment.from_file(os.path.join(folder_c, files_c[i]))

            # Make all sounds the same length as the longest one
            max_length = max(len(a), len(b), len(c))
            a = a + AudioSegment.silent(duration=max_length - len(a))
            b = b + AudioSegment.silent(duration=max_length - len(b))
            c = c + AudioSegment.silent(duration=max_length - len(c))

            layered = a.overlay(b).overlay(c)
            layered = layered.normalize(0)

            out_filename = f"layered_{i+1}.wav"
            layered.export(os.path.join(output_folder, out_filename), format="wav")
            print(f"Exported: {out_filename}")

        except Exception as e:
            print(f"Error processing set {i+1}: {e}")

    print("All sounds layered and exported.")


if __name__ == "__main__":
    layer_audio_from_folders(
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body",
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\fundamental",
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\transient",
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\layered",
    )
