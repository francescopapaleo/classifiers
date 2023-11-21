import soundfile as sf
import resampy
import os
import sys
from pathlib import Path


def resample_folder(file_path, target_fs):
    if not os.path.exists(file_path) or not os.path.isdir(file_path):
        print(f"Directory '{file_path}' does not exist or is not a directory.")
        return

    file_list = os.listdir(file_path)
    print(f"Files in directory: {file_path}")

    for file_name in file_list:
        single_file_path = Path(file_path) / file_name
        if single_file_path.suffix.lower() == ".wav":
            waveform, fs = sf.read(single_file_path)

            print(f"Processing file: {single_file_path}")
            print(f" - Dtype: {waveform.dtype}, Shape: {waveform.shape}")
            print(f" - Max:     {waveform.max().item():6.3f}")
            print(f" - Min:     {waveform.min().item():6.3f}")
            print(f" - Mean:    {waveform.mean().item():6.3f}")
            print(f" - Std Dev: {waveform.std().item():6.3f}")

            resampled = resampy.resample(waveform, fs, target_fs)
            os.makedirs(f"{file_path}/resampled", exist_ok=True)

            tag = int(target_fs / 1000)
            resampled_filename = f"{file_name}_{tag}k24b.wav"
            resampled_file_path = Path(file_path) / "resampled" / resampled_filename
            sf.write(resampled_file_path, resampled, target_fs, subtype="PCM_24")

            print(f" - Resampled file saved to: {resampled_file_path}", "\n")

    print("Done")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <target_sampling_rate>")
    else:
        input_directory = sys.argv[1]
        target_sampling_rate = int(sys.argv[2])
        resample_folder(file_path=input_directory, target_fs=target_sampling_rate)
