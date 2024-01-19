import sys
import os
import torch
import torchaudio
import pyloudnorm as pyln
from utils import find_audio


LEVEL = -14.0  # Target loudness level in LUFS
AUDIO_EXTENSION = ".wav"


def loudness_normalize(
    input_file_path: str,
    output_dir: str,
    level: float = LEVEL
):
    """
    Normalize audio to a target loudness level in LUFS.

    :param input_file_path: The path to the input audio file.
    :type input_file_path: str
    :param output_dir: The directory where the normalized file will be saved.
    :type output_dir: str
    :param level: The target loudness level in LUFS. Defaults to LEVEL.
    :type level: float
    """
    input_signal, sample_rate = torchaudio.load(input_file_path)
    input_metadata = torchaudio.info(input_file_path)
    print(input_metadata)
    input_signal = input_signal.squeeze(0).numpy()
    # Input data must have shape (samples, ch) or (samples,) for mono audio.

    # remove DC offset
    input_signal = input_signal - input_signal.mean()
    # peak normalize to -1 dBFS
    input_signal = pyln.normalize.peak(input_signal, -1.0)
    meter = pyln.Meter(sample_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(
        input_signal
    )  # measure loudness for the first channel
    print(f"Input loudness: {loudness} LUFS")

    output_signal = pyln.normalize.loudness(
        input_signal, loudness, level
    )  # Normalize first channel

    output_signal = torch.tensor(
        output_signal.reshape(1, -1)
    )  # Add channel dimension

    os.makedirs(output_dir, exist_ok=True)
    output_file_name = os.path.basename(
        input_file_path)[:-4] + f"-{abs(LEVEL)}LUFS.wav"
    output_file_path = os.path.join(output_dir, output_file_name)
    torchaudio.save(output_file_path, output_signal, sample_rate)

    output_loudness = meter.integrated_loudness(
        output_signal.squeeze(0).numpy())
    # output_loudness = torchaudio.functional.loudness(
    #   output_signal, sample_rate)  # Check loudness
    print(
        f"Output loudness: {output_loudness} LUFS", "\n"
        )


def main():
    try:
        indir = sys.argv[1]
    except IndexError:
        print("Usage:", sys.argv[1], "<input-directory> ")
        sys.exit(1)

    files = [f for f in find_audio(indir, AUDIO_EXTENSION)]

    print("Found", len(files), "audio files (" + AUDIO_EXTENSION + ")")

    output_dir = os.path.join(indir, "normalized")

    for i, path in enumerate(files, start=1):
        print("Normalizing:", path)
        try:
            loudness_normalize(path, output_dir=output_dir, level=LEVEL)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")


if __name__ == "__main__":
    main()
