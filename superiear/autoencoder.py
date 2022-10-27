import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import random

NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 128


class AudioDataset(Dataset):
    # see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def __len__(self):
        return len([name for name in os.listdir(self.processed_path) if name.endswith(".wav")])

    def __getitem__(self, idx):
        # get random file from processed_path
        # get corresponding file from raw_path
        processed_file = random.choice(os.listdir(self.processed_path))
        # deleted _<number>.wav from end of filename using regex
        raw_file = re.sub(r'_\d+.wav', ".wav", processed_file)
        print(f"raw_file: {raw_file}")
        print(f"processed_file: {processed_file}")
        processed = torch.load(os.path.join(
            self.processed_path, processed_file))
        raw = torch.load(os.path.join(self.raw_path, raw_file))
        # TODO: process both files to be tensors
        return {'processed': processed, 'original': raw}


trainset = AudioDataset(
    raw_path="data/raw",
    processed_path="data/processed"
)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


class DeepAutoencoder(nn.Module):
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
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


device = get_device()
DAE = DeepAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(DAE.parameters(), lr=LEARNING_RATE)


def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            d, _ = data
            original = d['original'].to(device)
            processed = d['processed'].to(device)
            optimizer.zero_grad()
            outputs = net(processed)
            loss = criterion(outputs, original)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {}. Train Loss: {:.3f}'.format(
            epoch+1, NUM_EPOCHS, loss))
        if epoch % 10 == 0:
            torch.save(net.state_dict(),
                       'DAE-checkpoints/DAE-epoch{}.pth'.format(epoch))
    return train_loss


def test(net, testloader):
    for batch in testloader:
        d, _ = batch
        processed = d['processed'].to(device)
        outputs = net(processed)
        # TODO: save output noises
        break


def make_dir():
    checkpoint_dir = 'DAE-checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


make_dir()
train_loss = train(DAE, trainloader, NUM_EPOCHS)
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png')