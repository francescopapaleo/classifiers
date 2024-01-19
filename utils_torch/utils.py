import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from typing import List


AUDIO_EXTENSIONS = [".wav", ".flac", ".aiff"]


def find_audio(folder: str, ext: List[str] = AUDIO_EXTENSIONS):
    """Finds all audio files in a directory recursively.
    Returns a list.

    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    """
    folder = Path(folder)
    # Take care of case where user has passed in an audio file directly
    # into one of the calling functions.
    if str(folder).endswith(tuple(ext)):
        # if, however, there's a glob in the path, we need to
        # return the glob, not the file.
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        files += folder.glob(f"**/*{x}")
    return files


def plot_waveform(waveform, sample_rate, filename=""):
    """From https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.savefig(filename + "waveform.png")


def plot_specgram(waveform, sample_rate, title="Spectrogram", filename=""):
    """From https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.savefig(filename + "spectrogram.png")


if __name__ == '__main__':

    try:
        indir = sys.argv[1]
    except IndexError:
        print("Usage:", sys.argv[1], "<input-directory> ")
        sys.exit(1)

    files = [f for f in find_audio(indir, AUDIO_EXTENSIONS)]
    print('Found', len(files),
          'audio files (' + ', '.join(AUDIO_EXTENSIONS) + ')')

    output_dir = os.path.join(indir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(files, start=1):
        print('Processing:', path)
        try:
            waveform, sample_rate = torchaudio.load(path)
            base_filename = os.path.splitext(os.path.basename(path))[0]
            plot_filename = os.path.join(output_dir, base_filename)
            plot_waveform(waveform, sample_rate, plot_filename)
            plot_specgram(waveform, sample_rate, filename=plot_filename)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
