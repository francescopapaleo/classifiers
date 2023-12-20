import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path


def resample_folder(dir_path, resample_sr) -> None:
    """
    Resamples all .wav files in a folder to a target sampling rate.
    Saves the resampled files to a subfolder named 'resampled'.

    Args:
        file_path (str): Path to the directory containing the .wav files.
        target_fs (int): Target sampling rate in Hz.

    Returns:
        None
    """

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        print(f"Directory '{dir_path}' does not exist or is not a directory.")
        return

    file_list = os.listdir(dir_path)
    print(f"{len(dir_path)} files in directory")

    for file_name in file_list:
        single_file_path = Path(dir_path) / file_name
        if single_file_path.suffix.lower() == ".wav":

            # Load audio file and read metadata
            print(f"Processing file: {single_file_path}")
            metadata = torchaudio.info(single_file_path)
            print(metadata)
            bit_depth = metadata.bits_per_sample

            waveform, sr = torchaudio.load(single_file_path)

            # Resample
            resampler = T.Resample(
                orig_freq=sr, new_freq=resample_sr, dtype=waveform.dtype
            )
            resampled_waveform = resampler(waveform)
            print(resampled_waveform.shape)

            # Convert to mono with equal energy in both channels
            if resampled_waveform.shape[0] == 2:
                left_ch = resampled_waveform[0, :]
                right_ch = resampled_waveform[1, :]
                left_energy = torch.sqrt(torch.mean(
                    torch.square(resampled_waveform[0, :]), dim=0))
                right_energy = torch.sqrt(torch.mean(
                    torch.square(resampled_waveform[1, :]), dim=0))

                total_energy = left_energy + right_energy
                leff_coeff = left_energy / total_energy
                right_coeff = right_energy / total_energy
                mono_waveform = leff_coeff * left_ch + right_coeff * right_ch

            # Normalize to [-1, 1]
            norm_waveform = mono_waveform / torch.max(torch.abs(mono_waveform))

            tag = int(resample_sr / 1000)
            file_name, extension = os.path.splitext(file_name)
            resampled_name = f"{file_name}_{tag}kHz{bit_depth}b.wav"
            os.makedirs(f"{dir_path}/resampled", exist_ok=True)
            resampled_path = Path(dir_path) / "resampled" / resampled_name

            # Save audio file
            torchaudio.save(
                uri=resampled_path,
                src=norm_waveform,
                sample_rate=resample_sr,
                bits_per_sample=bit_depth,
            )

            print(f" - Resampled file saved to: {resampled_path}", "\n")

    print("Done")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python script.py <input_directory> <target_sampling_rate>"
            )
    else:
        dir_path = sys.argv[1]
        resample_sr = int(sys.argv[2])
        resample_folder(dir_path=dir_path, resample_sr=resample_sr)
