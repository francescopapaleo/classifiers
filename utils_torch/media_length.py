import os
import torchaudio
import argparse


def get_media_length_in_dir(file_path):
    """
    Function to get the length of audio files in a directory.
    Supported file types: .wav, .wave

    Parameters
    ----------
    file_path : str
        Path to the directory containing the audio files.
    """
    file_list = os.listdir(file_path)
    print(f"Files in directory: {file_path}")
    print("")

    for file_name in file_list:
        single_file_path = os.path.join(file_path, file_name)
        print(f"Processing file: {single_file_path}")

        if single_file_path.endswith((".wav", ".wave")):
            waveform, sample_rate = torchaudio.load(single_file_path)
            total_length_samples = waveform.shape[1]
            total_length_seconds = total_length_samples / sample_rate
            total_number_of_frames = total_length_samples
            minutes, seconds = divmod(total_length_seconds, 60)
            print(
                "Length in {}: {} minutes, {:.2f} seconds"
                .format(single_file_path, int(minutes), seconds)
            )
            print(
                "Number of elements in {}: {}"
                .format(single_file_path, total_number_of_frames)
            )

        else:
            print("Skipping {} (unsupported file type)"
                  .format(single_file_path))

        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the length of audio files in a directory."
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the directory containing the audio files."
    )
    args = parser.parse_args()

    get_media_length_in_dir(args.file_path)
