import torch
import numpy as np
import noisereduce as nr

FRAMERATE = int(16000)

def spectral_model(batch):
    return torch.tensor(np.vstack([nr.reduce_noise(y=y, sr=FRAMERATE, n_jobs=-1) for y in batch]))