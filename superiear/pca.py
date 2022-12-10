import matplotlib.pyplot as plt
from torch import pca_lowrank as t_pca
from torch import matmul

from autoencoder import *


def pca_analysis(data):
    components = 2
    U, S, V = t_pca(data)
    projections = matmul(data, V[:, :components])
    return projections


def create_embeddings(net, data, iter):
    processed = data['processed'].to(device)
    processed = processed.reshape(1, 1, BATCH_SIZE, -1)
    net(processed)
    dae_embeddings = net.get_embeddings()
    return dae_embeddings[iter].squeeze()


def plot_data(original, embeddings, iter):
    original_2d = pca_analysis(original).numpy()
    embeddings_2d = pca_analysis(embeddings).detach().numpy()
    plt.plot(list(original_2d[:, 0].squeeze()), list(original_2d[:, 1].squeeze()), 'b.')
    plt.plot(list(embeddings_2d[:, 0].squeeze()), list(embeddings_2d[:, 1].squeeze()), 'g.')
    plt.savefig(f"../data/pca_pngs/{iter}.png")
    plt.clf()


if __name__ == "__main__":
    torch.manual_seed(0)
    dataset = AudioDataset(
        raw_path="../data/clear_samples",
        processed_path="../data/noisy_samples",
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

    DAE = DeepConvAutoencoder().to(device) #load in dae
    DAE.to(device)
    DAE.load_state_dict(torch.load("../models/dae_950.pth", map_location="cpu"))

    iter = 0
    for data in valloader:
        processed = data['processed'].to(device)
        embeddings = create_embeddings(DAE, data, iter)
        if embeddings is None:
            print("none")
            continue
        plot_data(processed, embeddings, iter)
        print(iter)
        iter += 1

