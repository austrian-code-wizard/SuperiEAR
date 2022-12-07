
import glob
import torch
import torchaudio
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    # see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.files = [f.split("/")[-1]
                      for f in glob.glob(f"{processed_path}/*.wav")]

        raw_files = [f.split("/")[-1]
                     for f in glob.glob(f"{raw_path}/*.wav")]
        assert all(
            f in raw_files for f in self.files), "Some processed files are not in the raw file folder"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # get random file from processed_path
        # get corresponding file from raw_path
        file = self.files[idx]
        raw, _ = torchaudio.load(f"{self.raw_path}/{file}")
        raw = torch.mean(raw, dim=0)
        raw = raw[raw.shape[0]//2:]
        processed, _ = torchaudio.load(f"{self.processed_path}/{file}")
        processed = torch.mean(processed, dim=0)
        processed = processed[processed.shape[0]//2:]
        assert raw.shape == processed.shape, f"Raw and processed shapes do not match: {raw.shape} vs {processed.shape}"
        return {'processed': processed, 'original': raw, 'filename': file}
