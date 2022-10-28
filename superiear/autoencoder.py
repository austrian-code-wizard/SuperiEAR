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
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter


import time
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
DECAY = 0.99
BATCH_SIZE = 128
TRACK_LENGTH = 7
FRAMERATE = 44100

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


sizes = [1024, 512, 256, 256]


class DeepAutoencoder(nn.Module):
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(TRACK_LENGTH * FRAMERATE, sizes[0]),
            nn.ReLU(True),
            nn.Linear(sizes[0], sizes[1]),
            nn.ReLU(True),
            nn.Linear(sizes[1], sizes[2]),
            nn.ReLU(True),
            nn.Linear(sizes[2], sizes[3]),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(sizes[3], sizes[2]),
            nn.ReLU(True),
            nn.Linear(sizes[2], sizes[1]),
            nn.ReLU(True),
            nn.Linear(sizes[1], sizes[0]),
            nn.ReLU(True),
            nn.Linear(sizes[0], TRACK_LENGTH * FRAMERATE),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def draw_map(net):
    x = torch.randn(1, TRACK_LENGTH * FRAMERATE)
    y = net(x)
    g = make_dot(y)
    g.render('DAE', view=False)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


writer = SummaryWriter()


def train(net, trainloader, valloader, valset, NUM_EPOCHS, device, criterion, optimizer):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
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
            writer.add_scalar('Loss/train', loss.item(), epoch)
        loss = running_loss / len(trainloader)
        writer.add_scalar('Loss/epoch', loss, epoch)
        train_loss.append(loss)
        print('Epoch {}. Train Loss: {:.3f} Time: {:.3f}'.format(
            epoch+1, loss, time.time() - t0))
        torch.save(net.state_dict(), f'./models/dae_{epoch}.pth')
        evaluate(net, valloader, valset, criterion, device, epoch)
        torch.save(net.state_dict(),
                   'DAE-checkpoints/DAE-epoch{}.pth'.format(epoch))
        test(net, valloader, device)
    return train_loss


def save_test_example(epoch, net, valset, device, output_dir="./val_examples"):
    make_dir(output_dir)
    output_dir = os.path.join(output_dir, f"epoch_{epoch}")
    make_dir(output_dir)
    val_examples = [random.randint(0, len(valset)) for _ in range(10)]
    with torch.no_grad():
        for idx in val_examples:
            try:
                data = valset[idx]
                original = data['original'].to(device)
                processed = data['processed'].to(device)
                outputs = net(processed)
                torchaudio.save(f"{output_dir}/{idx}_raw.wav",
                                original.reshape(1, -1), FRAMERATE)
                writer.add_audio(f"raw/{idx}", original, epoch)
                torchaudio.save(f"{output_dir}/{idx}_noisy.wav",
                                processed.reshape(1, -1), FRAMERATE)
                writer.add_audio(f"noisy/{idx}", processed, epoch)
                torchaudio.save(f"{output_dir}/{idx}_restored.wav",
                                outputs.reshape(1, -1), FRAMERATE)
                writer.add_audio(f"restored/{idx}", outputs, epoch)
            except Exception as e:
                print(e)
    pass


def evaluate(net, valloader, valset, criterion, device, epoch):
    net.eval()
    running_loss = 0.0
    for data in valloader:
        original = data['original'].to(device)
        processed = data['processed'].to(device)
        outputs = net(processed)
        loss = criterion(outputs, original)
        running_loss += loss.item()
    loss = running_loss / len(valloader)
    print('Validation Loss: {:.3f}'.format(loss))
    writer.add_scalar('Loss/validation', loss, epoch)
    net.train()
    save_test_example(epoch, net, valset, device)


def test(net, valloader, device):
    val_loss = 0.0
    with torch.no_grad():
        for data in valloader:
            original = data['original'].to(device)
            processed = data['processed'].to(device)
            outputs = net(processed)
            # save sample
            torchaudio.save('DAE-checkpoints/sample.wav',
                            outputs.cpu(), FRAMERATE)
            loss = criterion(outputs, original)
            val_loss += loss.item()
    val_loss /= len(valloader)
    print('Val Loss: {:.3f}'.format(loss))


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    make_dir("DAE-checkpoints")
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
        shuffle=True
    )

    device = get_device()
    DAE = DeepAutoencoder().to(device)
    draw_map(DAE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        DAE.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    train(DAE, trainloader, valloader, valset,
          NUM_EPOCHS, device, criterion, optimizer)
