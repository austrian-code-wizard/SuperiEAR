import matplotlib
import os
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio, display
from superiear.utils import ensure_dir_exists

PATHS = ["clear_samples", "noisy_samples", "final"]
# Adjusted from https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
# Visualize the waveform and spectrogram for a .wav file.
# Could use with original, noisy, and denoised .wav file

# give a filename which exists in final, clear, and noisy, and it save plots for all three
def create_plots(file_name):
    for path in PATHS:
        directory = f"../data/{path}/{file_name}.wav"
        if not os.path.exists(directory):
            print(f"directory {directory} does not exist, exiting creating plots")

    for path in PATHS:
        directory = f"../data/{path}/{file_name}.wav"
        spectrogram_waveform(directory, path.split("_")[0])

def spectrogram_waveform(file_path, audio_type):
    waveform, sample_rate = torchaudio.load(file_path)
    file_name = f"{audio_type}_" + file_path.split("/")[-1].split('.')[0]
    plot_waveform(waveform, sample_rate, "../data/waveforms", filename=f"{file_name}_waveform.png")
    plot_specgram(waveform, sample_rate, "../data/spectrograms", filename=f"{file_name}_specgram.png")

@ensure_dir_exists
def plot_specgram(waveform, sample_rate, spec_dir, title="Spectrogram", xlim=None, filename=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    if filename:
        plt.savefig(f"{spec_dir}/{filename}")
    else:
        plt.show(block=False)

@ensure_dir_exists
def plot_waveform(waveform, sample_rate, spec_dir, title="Waveform", xlim=None, ylim=None, filename=None):
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
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    if filename:
        plt.savefig(f"{spec_dir}/{filename}")
    else:
        plt.show(block=False)


if __name__ == "__main__":
    # waveform, sample_rate = torchaudio.load("../data/clear_samples/bob_ross_0.wav")
    create_plots("bob_ross_0")