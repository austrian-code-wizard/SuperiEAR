import time
from dataset import AudioDataset
from stft_loss import MultiResolutionSTFTLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from math import sqrt

NUM_EPOCHS = 1000 # change to 1000
LEARNING_RATE = 2e-4
DECAY = 0.999
BATCH_SIZE = 8

PICKUP_EPOCH = 0

TRAIN_NOISE_SCHEDULE = np.linspace(1e-4, 0.05, 50)
INFERENCE_NOISE_SCHEDULE = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])

RESIDUAL_CHANNELS = 64
RESIDUAL_LAYERS = 32
DILATION_CYCLE_LENGTH = 10

random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Upsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(1, 1, [3, 32], stride=[
            1, 16], padding=[1, 8])
        self.conv2 = nn.ConvTranspose2d(1, 1,  [3, 32], stride=[
            1, 16], padding=[1, 8])

    def forward(self, x):
        # print(x.shape)
        # x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        '''
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(
            1, 2 * residual_channels, 1)
        self.output_projection = Conv1d(
            residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(
            diffusion_step).unsqueeze(-1)

        y = x + diffusion_step

        conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, noise_schedule):
        super().__init__()
        self.input_projection = Conv1d(1, RESIDUAL_CHANNELS, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            len(noise_schedule))
        self.noisy_audio_upsampler = Upsampler()

        self.residual_layers = nn.ModuleList([
            ResidualBlock(RESIDUAL_CHANNELS, 2**(i %
                                                 DILATION_CYCLE_LENGTH))
            for i in range(RESIDUAL_LAYERS)
        ])
        self.skip_projection = Conv1d(
            RESIDUAL_CHANNELS, RESIDUAL_CHANNELS, 1)
        self.output_projection = Conv1d(RESIDUAL_CHANNELS, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, noisy_audio=None):
        assert (noisy_audio is None and self.noisy_audio_upsampler is None) or \
               (noisy_audio is not None and self.noisy_audio_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, noisy_audio)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


print("Running from device: ", device)
writer = SummaryWriter()


mrstftloss = MultiResolutionSTFTLoss(factor_sc=0.5,
                                     factor_mag=0.5)


def train_diffusion(net, trainloader, valloader, start_epoch, NUM_EPOCHS, criterion, optimizer):
    beta = np.array(TRAIN_NOISE_SCHEDULE)
    noise_level = torch.tensor(np.cumprod(1 - beta).astype(np.float32))
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    train_loss = []
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        running_loss = 0.0
        steps = 0
        for data in trainloader:
            original = data['original']
            processed = data['processed']

            if processed.shape[0] != BATCH_SIZE:
                continue
            original = original.reshape(BATCH_SIZE, 1, -1).to(device)
            processed = processed.reshape(BATCH_SIZE, 1, -1).to(device)
            t = torch.randint(0, len(TRAIN_NOISE_SCHEDULE), [1], device=device)
            noise_scale = noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale**0.5
            noise = torch.randn_like(original)
            noisy_audio = noise_scale_sqrt * original + \
                (1.0 - noise_scale)**0.5 * noise
            optimizer.zero_grad()
            net.to(device)
            outputs = net(noisy_audio, t, processed)
            loss = criterion(noise, outputs)
            writer.add_scalar('OG_loss/train', loss.item(), epoch)
            #sc_loss, mag_loss = mrstftloss(
            #    noisy_audio.squeeze(1) - outputs.squeeze(1), original.squeeze(1))
            #loss += sc_loss + mag_loss
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch)
            #writer.add_scalar('Loss/sc_loss', sc_loss.item(), epoch)
            #writer.add_scalar('Loss/mag_loss', mag_loss.item(), epoch)
            #writer.add_scalar('Loss/stfs_total',
            #                  mag_loss.item() + sc_loss.item(), epoch)
            if steps % 100 == 0:
                print(f'Epoch: {epoch} | Step: {steps} | Loss: {loss.item()}')
            steps += 1
        loss = running_loss / len(trainloader)
        writer.add_scalar('Loss/epoch', loss, epoch)
        train_loss.append(loss)
        print('Epoch {}. Train Loss: {:.3f} Time: {:.3f}'.format(
            epoch, loss, time.time() - t0))
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'./models/diffusion_{epoch}.pth')
        evaluate(net, valloader, criterion, epoch)
    return train_loss


def evaluate(net, valloader, criterion, epoch):
    net.diffusion_embedding.register_buffer('embedding', net.diffusion_embedding._build_embedding(
        len(INFERENCE_NOISE_SCHEDULE)), persistent=False)
    net.eval()
    running_loss = 0.0

    for data in valloader:
        original = data['original'].to(device)
        processed = data['processed'].to(device)
        if processed.shape[0] != BATCH_SIZE:
            continue
        original = original.reshape(BATCH_SIZE, 1, -1).to(device)
        processed = processed.reshape(BATCH_SIZE, 1, -1).to(device)
        outputs = infer(net, processed)
        loss = criterion(outputs, original)
        writer.add_scalar('OG_loss/val', loss.item(), epoch)
        #sc_loss, mag_loss = mrstftloss(
        #    outputs.squeeze(1), original.squeeze(1))
        #loss += sc_loss + mag_loss
        running_loss += loss.item()
        writer.add_scalar('Loss/val', loss.item(), epoch)
        #writer.add_scalar('Loss/sc_loss_val', sc_loss.item(), epoch)
        #writer.add_scalar('Loss/mag_loss_val', mag_loss.item(), epoch)
        #writer.add_scalar('Loss/stfs_total_val',
        #                  mag_loss.item() + sc_loss.item(), epoch)
    loss = running_loss / len(valloader)
    print('Validation Loss: {:.3f}'.format(loss))
    writer.add_scalar('Loss/validation', loss, epoch)
    net.diffusion_embedding.register_buffer('embedding', net.diffusion_embedding._build_embedding(
        len(TRAIN_NOISE_SCHEDULE)), persistent=False)
    net.train()


def infer(net, processed):
    beta = INFERENCE_NOISE_SCHEDULE
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = np.array(INFERENCE_NOISE_SCHEDULE, dtype=np.float32)
    audio = torch.randn(
        processed.shape[0],
        1,
        processed.shape[-1],
        device=device)

    with torch.no_grad():
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            audio = c1 * (audio - c2 * net(audio,
                        torch.tensor([T[n]], device=audio.device), processed))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) /
                        (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
        audio = torch.clamp(audio, -1.0, 1.0)
    return audio

def diffusion_model(path):
    net = DiffWave(INFERENCE_NOISE_SCHEDULE).to(device)
    net.to(device)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    dataset = AudioDataset(
        raw_path="data/clear_samples",
        processed_path="data/noisy_samples",
    )
    dataset.files = dataset.files[:10]

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

    Diff = DiffWave(TRAIN_NOISE_SCHEDULE).to(device)
    Diff.to(device)
    # print(Diff)
    if PICKUP_EPOCH:
        Diff.load_state_dict(torch.load(
            f"./models/dae_{PICKUP_EPOCH}.pth"))
    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        Diff.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)

    train_diffusion(Diff, trainloader, valloader,
                    PICKUP_EPOCH, NUM_EPOCHS, criterion, optimizer)
