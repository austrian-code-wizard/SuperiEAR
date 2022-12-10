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
BATCH_SIZE = 32
TRACK_LENGTH = 7
FRAMERATE = int(16000)

PICKUP_EPOCH = None

EVAL_EVERY = 1

SAMPLES_DIR = "samples"
SAVE_SAMPLES_EVERY = 1

CHECKPOINT_DIR = "models"
SAVE_MODEL_EVERY = 1


random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DeepConvAutoencoder(nn.Module):
    def __init__(self, chnls_in=1, chnls_out=1):
        super(DeepConvAutoencoder, self).__init__()
        self.down_conv_layer_1 = DownConvBlock(
            chnls_in, 1, norm=False, kernel_size=3)
        self.down_conv_layer_2 = DownConvBlock(1, 128, kernel_size=3)
        self.down_conv_layer_3 = DownConvBlock(128, 256, kernel_size=3)
        self.down_conv_layer_4 = DownConvBlock(
            256, 256, dropout=0.5, kernel_size=3)
        self.down_conv_layer_5 = DownConvBlock(
            256, 256, dropout=0.5, kernel_size=3)
        self.down_conv_layer_6 = DownConvBlock(
            256, 256, dropout=0.5, kernel_size=3)
        self.up_conv_layer_1 = UpConvBlock(
            256, 256, kernel_size=(2, 3), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_2 = UpConvBlock(512, 256, kernel_size=(
            2, 3), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_3 = UpConvBlock(512, 256, kernel_size=(
            2, 3), stride=2, padding=0, dropout=0.5)
        self.up_conv_layer_4 = UpConvBlock(512, 128, dropout=0.5)
        self.up_conv_layer_5 = UpConvBlock(256, 128)
        self.upsample_layer = nn.Upsample(
            scale_factor=(1, 2))
        self.conv_layer_1 = nn.Conv2d(128, chnls_out, 3, padding=1)
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

        out = self.upsample_layer(dec5)
        out = out[:, :128, :, :]  # ensure that the tensor has 128 channels
        out = self.conv_layer_1(out)
        return out

# reference: https://medium.com/@sriskandaryan/autoencoders-demystified-audio-signal-denoising-32a491ab023a

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


def reshape_audio(audio):
    audio = audio.reshape(audio.shape[0], 1, 1, -1)
    return audio


def train(net, trainloader, valloader, start_epoch, NUM_EPOCHS, criterion, optimizer):
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        running_loss = 0.0
        for data in trainloader:
            original = reshape_audio(data['original'])
            processed = reshape_audio(data['processed'])
            optimizer.zero_grad()
            net.to(device)
            outputs = net(processed)
            loss = criterion(outputs, original)
            writer.add_scalar('Train/L1_loss', loss.item(), epoch)
            sc_loss, mag_loss = mrstftloss(
                outputs.squeeze(1), original.squeeze(1))
            loss += sc_loss + mag_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('Loss/Train/combined_loss', loss.item(), epoch)
            writer.add_scalar('Loss/Train/sc_loss', sc_loss.item(), epoch)
            writer.add_scalar('Loss/Train/mag_loss', mag_loss.item(), epoch)
            writer.add_scalar('Loss/Train/stfs_total',
                              mag_loss.item() + sc_loss.item(), epoch)
        loss = running_loss / len(trainloader)
        print('Epoch {}. Train Loss: {:.3f} Time: {:.3f}'.format(
            epoch, loss, time.time() - t0))
        if epoch % SAVE_MODEL_EVERY == 0:
            torch.save(net.state_dict(), f'./{CHECKPOINT_DIR}/dae_{epoch}.pth')
        if epoch % EVAL_EVERY == 0:
            evaluate(net, valloader, criterion, epoch)
        if epoch % SAVE_SAMPLES_EVERY == 0:
            save_test_example(epoch, net, valloader)


def save_test_example(epoch, net, valloader, num_examples=5):
    make_dir(SAMPLES_DIR)
    output_dir = os.path.join(SAMPLES_DIR, f"epoch_{epoch}")
    make_dir(output_dir)
    with torch.no_grad():
        for i, data in enumerate(valloader):
            original = reshape_audio(data['original'])
            processed = reshape_audio(data['processed'])
            outputs = net(processed)
            for j in range(num_examples if num_examples < len(original) else len(original)):
                orignal_out = original[j].reshape(1, -1).cpu()
                processed_out = processed[j][0][0].reshape(1, -1).cpu()
                output_out = outputs[j][0][0].reshape(1, -1).cpu()
                torchaudio.save(os.path.join(
                    output_dir, f"original_{i}_{j}.wav"), orignal_out, FRAMERATE)
                torchaudio.save(os.path.join(
                    output_dir, f"processed_{i}_{j}.wav"), processed_out, FRAMERATE)
                torchaudio.save(os.path.join(
                    output_dir, f"output_{i}_{j}.wav"), output_out, FRAMERATE)
            if i == 0:
                break


def evaluate(net, valloader, criterion, epoch):
    net.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in valloader:
            original = reshape_audio(data['original'])
            processed = reshape_audio(data['processed'])
            outputs = net(processed)
            loss = criterion(outputs, original)
            writer.add_scalar('Loss/Val/L1_Loss', loss.item(), epoch)
            sc_loss, mag_loss = mrstftloss(
                outputs.squeeze(1), original.squeeze(1))
            loss += sc_loss + mag_loss
            running_loss += loss.item()
            writer.add_scalar('Loss/Val/combined_loss', loss.item(), epoch)
            writer.add_scalar('Loss/Val/sc_loss', sc_loss.item(), epoch)
            writer.add_scalar('Loss/Val/mag_loss', mag_loss.item(), epoch)
            writer.add_scalar('Loss/Val/stfs_total',
                              mag_loss.item() + sc_loss.item(), epoch)
    loss = running_loss / len(valloader)
    print(f"Epoch {epoch}. Val Loss: {loss}")
    net.train()


def deep_conv_autoencoder_model(path):
    model = DeepConvAutoencoder().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


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
    print(f"Validation dataset size: {len(valset)}")

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
    if PICKUP_EPOCH is not None:
        DAE.load_state_dict(torch.load(
            f"./{CHECKPOINT_DIR}/dae_{PICKUP_EPOCH}.pth"))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        DAE.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    data = next(iter(trainloader))

    train(DAE, trainloader, valloader, 1 if PICKUP_EPOCH is None else PICKUP_EPOCH,
          NUM_EPOCHS, criterion, optimizer)

