import torch;# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
device = 'cuda' if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(OrderedDict([('flatten', nn.Flatten(start_dim=1)),
                                  ('linear1', nn.Linear(784,512)),
                                  ('relu1',  nn.Sigmoid()),
                                  ('linear2', nn.Linear(512, latent_dims))]))

    def forward(self, x):
        return self.net(x)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))

        self.linear2 = nn.Linear(64*20*20, latent_dims)
        self.linear3 = nn.Linear(64*20*20, latent_dims)

        self.N = torch.distributions.Normal(0,1)
        self.N.loc  = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        #self.kl = (sigma **2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = (1 + torch.log(sigma) - mu**2 - torch.exp(torch.log(sigma))**2).sum() * -0.5
        return z




class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.conv1_t = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5)
        self.conv2_t = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5)
        self.linear2 = nn.Linear(latent_dims, 64 * 20 * 20 )

    def forward(self, z):
        z = self.linear2(z)
        z = F.relu(z)
        z = z.reshape((-1,64,20,20))
        z = self.conv2_t(z)
        z = F.relu(z)
        z = self.conv1_t(z)
        z = F.relu(z)
        return z


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)
from tqdm import tqdm

LATENT_DIMS = 10
SNAPSHOT = "../snapshots/snapshot.pth"
def train(autoencoder, data, epochs, tb_writer):
    opt = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    counter = 0
    for epoch in range(epochs):
        with tqdm(data, unit="batch") as tepoch:
            for x,_ in tepoch:
                x = x.to(device)
                tepoch.set_description(f"Epoch {epoch}")
                opt.zero_grad()
                x_hat = autoencoder(x)
                loss = ((x-x_hat)**2).sum() + autoencoder.encoder.kl
                tepoch.set_postfix(loss=loss.item())
                loss.backward()
                opt.step()
        if tb_writer:
            tb_writer.add_scalar('Loss/train', loss.item(), epoch)
        images = create_recons(autoencoder, 10, latent_dims=LATENT_DIMS)
        fig_gen = plot_image_list(images)
        tb_writer.add_figure("reconstruction", fig_gen, global_step=epoch)
    return autoencoder

def plot_latent(autoencoder, data, num_batches=100):
    for b_idx, (x,y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z_display = z.to('cpu').detach().numpy()
        plt.scatter(z_display[:,0], z_display[:,1], c=y, cmap="tab10")
        if b_idx > num_batches:
            plt.colorbar()
            break
    #plt.show()
#https://arxiv.org/pdf/1906.02691.pdf
def plot_generated(autoencoder, r0=(-2,2), r1=(-2, 2), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    fig = plt.figure()
    plt.imshow(img, extent=[*r0, *r1])
    return fig

def create_recons(autoencoder, count, latent_dims):
    z = torch.distributions.Normal(0,1).sample((count, latent_dims)).to('cuda')
    #z = torch.rand(count, latent_dims).to('cuda')
    img = autoencoder.decoder(z)
    imgs = img.reshape(count, 28,28).detach().to('cpu').numpy()
    return imgs

from typing import List

def plot_image_list(images: List):
    n_col = 5
    n_row = int(len(images) / n_col)
    #fig = plt.figure()
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img)
        ax.axis('off')
    return fig

#https://avandekleut.github.io/vae/
import os
from torch.utils.tensorboard import SummaryWriter
def main():
    if not os.path.exists("../snapshots"):
       os.makedirs("../snapshots")
    autoencoder = Autoencoder(latent_dims=LATENT_DIMS).to(device)
    try:
        autoencoder.load_state_dict(torch.load(SNAPSHOT))
    except FileNotFoundError as err:
        pass
    data = torch.utils.data.DataLoader(torchvision.datasets.MNIST('/tmp', transform=torchvision.transforms.ToTensor(),
                                                                  download=True), batch_size=128, shuffle=True)

    tb_writer = SummaryWriter('../tensorboard/runs/mnist-vae-conv')
    autoencoder = train(autoencoder, data, epochs=2, tb_writer=tb_writer)
    torch.save(autoencoder.state_dict(),SNAPSHOT)
    plt.rcParams['figure.dpi'] = 200
    imgs = create_recons(autoencoder, 10, latent_dims=LATENT_DIMS)
    plot_image_list(imgs)
    plt.show()
    tb_writer.close()

if __name__ == '__main__':
    main()
