import os
import torch
import utils
import random
import torchaudio
import noisereduce as nr
from autoencoder import AudioDataset, FRAMERATE

@utils.ensure_dir_exists
def save_examples(output_dir):
  dataset = AudioDataset("./data/processed", "./data/final")
  random.seed(229)
  for i in random.sample(range(len(dataset)), 10):
    data = dataset[i]
    original = data["original"]
    processed = data["processed"]
    denoised = nr.reduce_noise(y=processed, sr=FRAMERATE, n_jobs=-1)
    for sample in zip([original, processed, denoised], ["clean", "noisy", "denoised"]):
      torchaudio.save(f"{output_dir}/{i}_{sample[1]}.wav", torch.tensor(sample[0]).reshape(1, -1), FRAMERATE)

if __name__ == "__main__":
  save_examples("./spectral_gating_examples")