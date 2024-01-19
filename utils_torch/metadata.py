import sys
import torch
import torchaudio
import yaml
from utils import find_audio
from pathlib import Path

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.ogg', '.flac', '.aiff', '.aif', '.aifc', '.m4a', '.mp4'
    ]


def index_metadata(folder):
    """Index metadata for audio files in a folder and write to a YAML file."""

    base_path = Path(folder)
    metadata_file = base_path / 'all_metadata.yaml'

    # Open the file and clear if it already exists
    with open(metadata_file, 'w') as file:
        pass

    for metadata_path in find_audio(folder, AUDIO_EXTENSIONS):
        relative_path = Path(metadata_path).relative_to(base_path)
        waveform, sample_rate = torchaudio.load(metadata_path)

        metadata_info = torchaudio.info(metadata_path)
        loudness = torchaudio.functional.loudness(waveform, sample_rate)
        mean = waveform.mean()
        std = waveform.std()
        rms = waveform.pow(2).mean().sqrt()
        rms_db = 20 * torch.log10(rms)

        metadata = {
            str(relative_path): {
                'sample_rate': sample_rate,
                'num_channels': waveform.size(0),
                'num_frames': waveform.size(1),
                'bits_per_sample': metadata_info.bits_per_sample,
                'loudness': loudness.item(),
                'mean': mean.item(),
                'std': std.item(),
                'rms': rms_db.item(),
            }
        }

        with open(metadata_file, 'a') as file:  # Append mode
            yaml.dump(metadata, file)


if __name__ == '__main__':
    try:
        indir = sys.argv[1]
    except IndexError:
        print(f"Usage: {sys.argv[0]} <input-directory>")
        sys.exit(1)

    index_metadata(indir)
    metadata_file = Path(indir) / 'all_metadata.yaml'
    print(f"Metadata for all files written to {metadata_file}")
