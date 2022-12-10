import os
import torch
import random
import torchaudio
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from stft_loss import MultiResolutionSTFTLoss
from dataset import AudioDataset
import time

NUM_EPOCHS = 1000
LEARNING_RATE = 3e-4
DECAY = 0.999
BATCH_SIZE = 64
TRACK_LENGTH = 7
FRAMERATE = int(16000)

PICKUP_EPOCH = 0

random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeepConvAutoencoder(nn.Module):
    def __init__(self, chnls_in=1, chnls_out=1):
        super(DeepConvAutoencoder, self).__init__()
        self.down_conv_layer_1 = DownConvBlock(chnls_in, 64, norm=False)
        self.down_conv_layer_2 = DownConvBlock(64, 128)
        self.down_conv_layer_3 = DownConvBlock(128, 256)
        self.down_conv_layer_4 = DownConvBlock(256, 256, dropout=0.5)
        self.down_conv_layer_5 = DownConvBlock(256, 256, dropout=0.5)
        self.down_conv_layer_6 = DownConvBlock(256, 256, dropout=0.5)

        self.up_conv_layer_1 = UpConvBlock(
            256, 256, kernel_size=(2, 3), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_2 = UpConvBlock(512, 256, kernel_size=(
            2, 3), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_3 = UpConvBlock(512, 256, kernel_size=(
            2, 3), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_4 = UpConvBlock(512, 128, dropout=0.5)
        self.up_conv_layer_5 = UpConvBlock(256, 64)
        self.up_conv_layer_6 = UpConvBlock(512, 128)
        self.up_conv_layer_7 = UpConvBlock(256, 64)
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv_layer_1 = nn.Conv2d(128, chnls_out, 4, padding=1)
        self.activation = nn.Tanh()

        self.embeddings = []

    def forward(self, x):
        enc1 = self.down_conv_layer_1(x)
        enc2 = self.down_conv_layer_2(enc1)
        enc3 = self.down_conv_layer_3(enc2)
        enc4 = self.down_conv_layer_4(enc3)
        enc5 = self.down_conv_layer_5(enc4)
        enc6 = self.down_conv_layer_6(enc5)

        self.embeddings.append(enc6)

        dec1 = self.up_conv_layer_1(enc6, enc5)
        dec2 = self.up_conv_layer_2(dec1, enc4)
        dec3 = self.up_conv_layer_3(dec2, enc3)
        dec4 = self.up_conv_layer_4(dec3, enc2)
        dec5 = self.up_conv_layer_5(dec4, enc1)

        final = self.upsample_layer(dec5)
        final = self.zero_pad(final)
        final = self.conv_layer_1(final)
        return final

    def get_embeddings(self):
        return self.embeddings

class UpConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, stride=2, padding=1, dropout=0.0):
        super(UpConvBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(
                ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(op_sz),
            nn.ReLU(),
        ])
        if dropout:
            self.layers += [nn.Dropout(dropout)]

    def forward(self, x, enc_ip):
        x = nn.Sequential(*(self.layers))(x)
        if x.shape != enc_ip.shape:
            x = nn.functional.interpolate(x, size=enc_ip.shape[2:])
        op = torch.cat((x, enc_ip), 1)
        return op


class DownConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, norm=True, dropout=0.0):
        super(DownConvBlock, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(ip_sz, op_sz, kernel_size, 2, 1)])
        if norm:
            self.layers.append(nn.InstanceNorm2d(op_sz))
        self.layers += [nn.LeakyReLU(0.2)]
        if dropout:
            self.layers += [nn.Dropout(dropout)]

    def forward(self, x):
        op = nn.Sequential(*(self.layers))(x)
        return op


print("Running from device: ", device)
writer = SummaryWriter()


mrstftloss = MultiResolutionSTFTLoss(factor_sc=0.5,
                                     factor_mag=0.5)


def train(net, trainloader, valloader, valset, start_epoch, NUM_EPOCHS, criterion, optimizer):
    train_loss = []
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        running_loss = 0.0
        for data in trainloader:
            original = data['original']
            processed = data['processed']
            if processed.shape[0] != BATCH_SIZE:
                continue
            # reshape to (batch_size, 1, track_length, 1)
            original = original.reshape(1, 1, BATCH_SIZE, -1).to(device)
            processed = processed.reshape(1, 1, BATCH_SIZE, -1).to(device)
            optimizer.zero_grad()
            net.to(device)
            outputs = net(processed)
            loss = criterion(outputs, original)
            writer.add_scalar('OG_loss/train', loss.item(), epoch)
            sc_loss, mag_loss = mrstftloss(
                outputs.squeeze(1), original.squeeze(1))
            loss += sc_loss + mag_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Loss/sc_loss', sc_loss.item(), epoch)
            writer.add_scalar('Loss/mag_loss', mag_loss.item(), epoch)
            writer.add_scalar('Loss/stfs_total',
                              mag_loss.item() + sc_loss.item(), epoch)
        loss = running_loss / len(trainloader)
        writer.add_scalar('Loss/epoch', loss, epoch)
        train_loss.append(loss)
        print('Epoch {}. Train Loss: {:.3f} Time: {:.3f}'.format(
            epoch+1, loss, time.time() - t0))
        torch.save(net.state_dict(), f'./models/dae_{epoch}.pth')
        evaluate(net, valloader, valset, criterion, epoch)
        # test(net, valloader, device)
    return train_loss


def save_test_example(epoch, net, valloader, output_dir="./val_examples"):
    make_dir(output_dir)
    output_dir = os.path.join(output_dir, f"epoch_{epoch}")
    make_dir(output_dir)
    for i, data in enumerate(valloader):
        original = data['original'].to(device)
        processed = data['processed'].to(device)
        if processed.shape[0] != BATCH_SIZE:
            continue
        original = original.reshape(BATCH_SIZE, 1, -1)
        processed = processed.reshape(1, 1, BATCH_SIZE, -1)
        outputs = net(processed).to(device)
        for j in range(5):
            orignal_out = original[j].reshape(1, -1).cpu()
            processed_out = processed[0][0][j].reshape(1, -1).cpu()
            output_out = outputs[0][0][j].reshape(1, -1).cpu()
            torchaudio.save(os.path.join(
                output_dir, f"original_{i}_{j}.wav"), orignal_out, FRAMERATE)
            # writer.add_audio(f'original_{i}_{j}',o, epoch)
            torchaudio.save(os.path.join(
                output_dir, f"processed_{i}_{j}.wav"), processed_out, FRAMERATE)
            # writer.add_audio(f'processed_{i}_{j}', processed[j], epoch)
            torchaudio.save(os.path.join(
                output_dir, f"output_{i}_{j}.wav"), output_out, FRAMERATE)
            # writer.add_audio(f'output_{i}_{j}', outputs[j].cpu(), epoch)
            if i == 0:
                break


def evaluate(net, valloader, valset, criterion, epoch):
    net.eval()
    running_loss = 0.0
    for data in valloader:
        original = data['original'].to(device)
        processed = data['processed'].to(device)
        if processed.shape[0] != BATCH_SIZE:
            continue
        original = original.reshape(1, 1, BATCH_SIZE, -1)
        processed = processed.reshape(1, 1, BATCH_SIZE, -1)
        outputs = net(processed)
        loss = criterion(outputs, original)
        writer.add_scalar('OG_loss/val', loss.item(), epoch)
        sc_loss, mag_loss = mrstftloss(
            outputs.squeeze(1), original.squeeze(1))
        loss += sc_loss + mag_loss
        running_loss += loss.item()
        writer.add_scalar('Loss/val', loss.item(), epoch)
        writer.add_scalar('Loss/sc_loss_val', sc_loss.item(), epoch)
        writer.add_scalar('Loss/mag_loss_val', mag_loss.item(), epoch)
        writer.add_scalar('Loss/stfs_total_val',
                          mag_loss.item() + sc_loss.item(), epoch)
    loss = running_loss / len(valloader)
    print('Validation Loss: {:.3f}'.format(loss))
    writer.add_scalar('Loss/validation', loss, epoch)
    net.train()
    save_test_example(epoch, net, valloader)


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    dataset = AudioDataset(
        raw_path="data/clear_samples",
        processed_path="data/noisy_samples",
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

    DAE = DeepConvAutoencoder().to(device)
    DAE.to(device)
    print(DAE)
    if PICKUP_EPOCH:
        DAE.load_state_dict(torch.load(
            f"./models2/dae_{PICKUP_EPOCH}.pth"))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        DAE.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    data = next(iter(trainloader))

    train(DAE, trainloader, valloader, valset,
          PICKUP_EPOCH, NUM_EPOCHS, criterion, optimizer)