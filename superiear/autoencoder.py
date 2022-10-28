import os
import glob
import torch
import random
import torchaudio
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


import time
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
DECAY = 0.997
BATCH_SIZE = 128
TRACK_LENGTH = 7
FRAMERATE = int(44100)

PICKUP_EPOCH = 44

random.seed(0)


class AudioDataset(Dataset):
    # see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.files = [f.split("/")[-1]
                      for f in glob.glob(f"{processed_path}/*.wav")]

        raw_files = [f.split("/")[-1] for f in glob.glob(f"{raw_path}/*.wav")]
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
        processed, _ = torchaudio.load(f"{self.processed_path}/{file}")
        processed = torch.mean(processed, dim=0)
        assert raw.shape == processed.shape, f"Raw and processed shapes do not match: {raw.shape} vs {processed.shape}"
        return {'processed': processed, 'original': raw}


class DeepAutoencoder(nn.Module):
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


print("Running from device: ", get_device())
writer = SummaryWriter()


def train(net, trainloader, valloader, valset, start_epoch, NUM_EPOCHS, device, criterion, optimizer):
    train_loss = []
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        running_loss = 0.0
        for data in trainloader:
            original = data['original'].to(device)
            processed = data['processed'].to(device)
            if processed.shape[0] != BATCH_SIZE:
                continue
            # reshape to (batch_size, 1, track_length, 1)
            original = original.reshape(BATCH_SIZE, 1, -1, 1)
            processed = processed.reshape(BATCH_SIZE, 1, -1, 1)
            optimizer.zero_grad()
            outputs = net(processed)
            loss = criterion(outputs, original)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch)
        loss = running_loss / len(trainloader)
        writer.add_scalar('Loss/epoch', loss, epoch)
        train_loss.append(loss)
        print('Epoch {}. Train Loss: {:.3f} Time: {:.3f}'.format(
            epoch+1, loss, time.time() - t0))
        torch.save(net.state_dict(), f'./models/dae_{epoch}.pth')
        evaluate(net, valloader, valset, criterion, device, epoch)
        # test(net, valloader, device)
    return train_loss


def save_test_example(epoch, net, valloader, device, output_dir="./val_examples"):
    make_dir(output_dir)
    output_dir = os.path.join(output_dir, f"epoch_{epoch}")
    make_dir(output_dir)
    for i, data in enumerate(valloader):
        original = data['original'].to(device)
        processed = data['processed'].to(device)
        if processed.shape[0] != BATCH_SIZE:
            continue
        # reshape to (batch_size, 1, track_length, 1)
        original = original.reshape(BATCH_SIZE, 1, -1, 1).to(device)
        processed = processed.reshape(BATCH_SIZE, 1, -1, 1).to(device)
        outputs = net(processed).to(device)
        for j in range(5):
            torchaudio.save(os.path.join(
                output_dir, f"original_{i}_{j}.wav"), original[j].reshape(1, -1), FRAMERATE)
            writer.add_audio(f'original_{i}_{j}', original[j], epoch)
            torchaudio.save(os.path.join(
                output_dir, f"processed_{i}_{j}.wav"), processed[j].reshape(1, -1), FRAMERATE)
            writer.add_audio(f'processed_{i}_{j}', processed[j], epoch)
            torchaudio.save(os.path.join(
                output_dir, f"output_{i}_{j}.wav"), outputs[j].reshape(1, -1), FRAMERATE)
            writer.add_audio(f'output_{i}_{j}', outputs[j], epoch)
            if i == 0:
                break


def evaluate(net, valloader, valset, criterion, device, epoch):
    net.eval()
    running_loss = 0.0
    for data in valloader:
        original = data['original'].to(device)
        processed = data['processed'].to(device)
        if processed.shape[0] != BATCH_SIZE:
            continue
        original = original.reshape(BATCH_SIZE, 1, -1, 1)
        processed = processed.reshape(BATCH_SIZE, 1, -1, 1)
        outputs = net(processed)
        loss = criterion(outputs, original)
        running_loss += loss.item()
    loss = running_loss / len(valloader)
    print('Validation Loss: {:.3f}'.format(loss))
    writer.add_scalar('Loss/validation', loss, epoch)
    net.train()
    save_test_example(epoch, net, valloader, device)


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    dataset = AudioDataset(
        raw_path="data/processed",
        processed_path="data/final"
    )

    trainset, valset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    print(f"Train dataset size: {len(trainset)}")
    print(f"Val dataset size: {len(valset)}")

    trainloader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    valloader = DataLoader(
        valset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    device = get_device()
    # draw_map(DAE)
    DAE = DeepAutoencoder().to(device)
    if PICKUP_EPOCH > 0:
        DAE.load_state_dict(torch.load(
            f'./models/dae_{PICKUP_EPOCH}.pth'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        DAE.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    data = next(iter(trainloader))

    train(DAE, trainloader, valloader, valset,
          PICKUP_EPOCH, NUM_EPOCHS, device, criterion, optimizer)
