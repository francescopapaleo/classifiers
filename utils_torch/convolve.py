import torch
import torchaudio
import torchaudio.functional as F


def convolve_audio(in_waveform, impulse_response, method='time'):
    print("Input waveform shape:", in_waveform.shape)
    if method == 'time':
        conv_waveform = F.convolve(in_waveform, impulse_response)
    elif method == 'fft':
        conv_waveform = F.fftconvolve(in_waveform, impulse_response)
    else:
        raise ValueError(
            "Invalid convolution method. Use 'time' or 'fft'."
            )

    # Normalize the output waveform
    norm_waveform = conv_waveform / torch.max(torch.abs(conv_waveform))

    return norm_waveform


def main(in_file, out_file, ir_file, method='time'):
    # Load the input audio file
    waveform, sample_rate = torchaudio.load(in_file)

    # Load the impulse response
    impulse_response, _ = torchaudio.load(ir_file)

    # Ensure mono audio
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Apply convolution
    conv_waveform = convolve_audio(waveform, impulse_response, method)

    print("Convolution waveform shape:", conv_waveform.shape)

    # Save the resulting audio
    torchaudio.save(out_file, conv_waveform, sample_rate)


if __name__ == "__main__":
    in_file = "../audio/plk-fm-base.wav"
    out_file = "../output_convolve.wav"
    ir_file = "../ir/IR_AKG_BX25_1500ms_48kHz24b.wav"
    conv_method = "fft"  # or "fft"

    main(in_file, out_file, ir_file, conv_method)
