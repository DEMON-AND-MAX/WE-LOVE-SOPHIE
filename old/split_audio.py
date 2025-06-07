import os
from pydub import AudioSegment


def split_audio_folder(input_folder, output_folder, chunk_length_sec=30):
    """
    Splits all audio files in a folder into chunks and saves them to another folder.

    Parameters:
        input_folder (str): Path to the folder containing input audio files.
        output_folder (str): Path to the folder where the chunks will be saved.
        chunk_length_sec (int): Length of each chunk in seconds.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            continue  # skip non-audio files

        try:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Convert to 16kHz
            audio = audio.set_sample_width(2)  # Convert to 16-bit
        except Exception as e:
            print(f"Could not load {filename}: {e}")
            continue

        chunk_length_ms = chunk_length_sec * 1000
        total_chunks = len(audio) // chunk_length_ms + (
            1 if len(audio) % chunk_length_ms else 0
        )

        for i in range(total_chunks):
            start = i * chunk_length_ms
            end = start + chunk_length_ms
            chunk = audio[start:end]
            chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i+1}.wav"
            chunk.export(os.path.join(output_folder, chunk_filename), format="wav")
            print(f"Exported: {chunk_filename}")

    print("All files processed.")


if __name__ == "__main__":
    split_audio_folder(
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\fundamental_full",
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\fundamental",
        chunk_length_sec=1,
    )

    split_audio_folder(
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\transient_full",
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\transient",
        chunk_length_sec=1,
    )

    split_audio_folder(
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body_full",
        "C:\\Users\\cools\\Desktop\\datasets\\dataset_b\\body",
        chunk_length_sec=1,
    )
