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

NUM_EPOCHS = 4
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
TRACK_LENGTH = 7
FRAMERATE = 44100

random.seed(0)


class AudioDataset(Dataset):
    # see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.files = [f.split("/")[-1] for f in glob.glob(f"{processed_path}/*.wav")]

        raw_files = [f.split("/")[-1] for f in glob.glob(f"{raw_path}/*.wav")]
        assert all(f in raw_files for f in self.files), "Some processed files are not in the raw file folder"

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
        self.encoder = nn.Sequential(
            nn.Linear(TRACK_LENGTH * FRAMERATE, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, TRACK_LENGTH * FRAMERATE),
            nn.Sigmoid()
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


def train(net, trainloader, valloader, NUM_EPOCHS, device, criterion, optimizer):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            original = data['original'].to(device)
            processed = data['processed'].to(device)
            optimizer.zero_grad()
            outputs = net(processed)
            loss = criterion(outputs, original)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {}. Train Loss: {:.3f}'.format(
            epoch+1, loss))
        torch.save(net.state_dict(),
                    'DAE-checkpoints/DAE-epoch{}.pth'.format(epoch))
        test(net, valloader, device)
    return train_loss


def test(net, valloader, device):
    val_loss = 0.0
    with torch.no_grad():
        for data in valloader:
            original = data['original'].to(device)
            processed = data['processed'].to(device)
            outputs = net(processed)
            loss = criterion(outputs, original)
            val_loss += loss.item()
    val_loss /= len(valloader)
    print('Val Loss: {:.3f}'.format(loss))


def save_test_example(net, valset, device, output_dir="./val_examples"):
    make_dir(output_dir)
    val_examples = [random.randint(0, len(valset)) for _ in range(10)]
    with torch.no_grad():
        for idx in val_examples:
            data = valset[idx]
            original = data['original'].to(device)
            processed = data['processed'].to(device)
            outputs = net(processed)
            torchaudio.save(f"{output_dir}/{idx}_raw.wav", original.reshape(1, -1), FRAMERATE)
            torchaudio.save(f"{output_dir}/{idx}_noisy.wav", processed.reshape(1, -1), FRAMERATE)
            torchaudio.save(f"{output_dir}/{idx}_restored.wav", outputs.reshape(1, -1), FRAMERATE)
    pass


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if __name__ == "__main__":
    make_dir("DAE-checkpoints")
    dataset = AudioDataset(
        raw_path="data/processed",
        processed_path="data/final"
    )

    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
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
        shuffle=True
    )

    device = get_device()
    DAE = DeepAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(DAE.parameters(), lr=LEARNING_RATE)

    train_loss = train(DAE, trainloader, valloader, NUM_EPOCHS, device, criterion, optimizer)
    save_test_example(DAE, valset, device)
    train_loss = 0
    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')