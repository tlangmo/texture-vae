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

TEXTURE_SIZE = 128
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3*32, kernel_size=(4,4), stride=(2,2), padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3*32)
        self.conv2 = nn.Conv2d(in_channels=3*32, out_channels=3*64, kernel_size=(4, 4),stride=(2,2),padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3*64)
        self.conv3 = nn.Conv2d(in_channels=3 * 64, out_channels=3 * 128, kernel_size=(4, 4), stride=(2, 2), padding=1)
        self.bn3 = torch.nn.BatchNorm2d(3 * 128)

        self.linear2 = nn.Linear(3*128*TEXTURE_SIZE//8*TEXTURE_SIZE//8, latent_dims)
        self.linear3 = nn.Linear(3*128*TEXTURE_SIZE//8*TEXTURE_SIZE//8, latent_dims)
        #
        self.N = torch.distributions.Normal(0,1)
        self.N.loc  = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        # #self.kl = (sigma **2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = (1 + torch.log(sigma) - mu**2 - torch.exp(torch.log(sigma))**2).sum() * -0.5
        return z




class Decoder(nn.Module):

    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.conv1_t = nn.ConvTranspose2d(in_channels=3*32, out_channels=3, kernel_size=(4,4), stride=(2,2), padding=1)

        self.bn1 = torch.nn.BatchNorm2d(3)
        self.conv2_t = nn.ConvTranspose2d(in_channels=3*64, out_channels=3*32, kernel_size=(4,4), stride=(2,2), padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3 * 32)
        self.conv3_t = nn.ConvTranspose2d(in_channels=3 * 128, out_channels=3 * 64, kernel_size=(4, 4), stride=(2, 2),
                                          padding=1)
        self.bn3 = torch.nn.BatchNorm2d(3 * 64)
        self.linear2 = nn.Linear(latent_dims, 3*128*TEXTURE_SIZE//8*TEXTURE_SIZE//8)

    def forward(self, z):
        z = self.linear2(z)
        z = F.relu(z)
        z = z.reshape((-1,3*128,TEXTURE_SIZE//8,TEXTURE_SIZE//8))
        z = self.conv3_t(z)
        z = F.relu(self.bn3(z))
        z = self.conv2_t(z)
        z = F.relu(self.bn2(z))
        z = self.conv1_t(z)
        z = F.relu(self.bn1(z))
        return z

#https://github.com/podgorskiy/VAE/blob/master/VAE.py
#https://www.echevarria.io/blog/lego-face-vae/
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)

from tqdm import tqdm
from torchsummary import summary
LATENT_DIMS = 32
SNAPSHOT = "../snapshots/snapshot_texture_wed.pth"
def train(autoencoder, data, epochs, tb_writer):
    opt = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-5)
    counter = 0
    for epoch in range(epochs):
        with tqdm(data, unit="batch") as tepoch:
            for x in tepoch:
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
        if epoch % 10:
            torch.save(autoencoder.state_dict(), SNAPSHOT)
    return autoencoder


def create_recons(autoencoder, count, latent_dims):
    z = torch.distributions.Normal(0,1).sample((count, latent_dims)).to('cuda')
    img = autoencoder.decoder(z)
    imgs = (img.reshape(count, 3,TEXTURE_SIZE,TEXTURE_SIZE).detach().to('cpu').permute(0,2,3,1).numpy()*255).astype('uint8')
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
import texture_dataset
def main():
    if not os.path.exists("../snapshots"):
       os.makedirs("../snapshots")
    autoencoder = Autoencoder(latent_dims=LATENT_DIMS).to(device)
    try:
        autoencoder.load_state_dict(torch.load(SNAPSHOT))
    except FileNotFoundError as err:
        pass
    data = torch.utils.data.DataLoader(texture_dataset.TextureDataset('./crops', transform=torchvision.transforms.Resize(TEXTURE_SIZE)),
                                       batch_size=32,
                                       shuffle=False)
    summary(autoencoder, (3, TEXTURE_SIZE, TEXTURE_SIZE))
    input_images = next(iter(data))
    plot_image_list([(input_images[b].permute(1,2,0).numpy()*255).astype('uint8') for b in range(input_images.shape[0])])
    plt.show()
    plt.rcParams['figure.dpi'] = 300
    tb_writer = SummaryWriter('../tensorboard/runs/mnist-vae-conv-wed')
    autoencoder = train(autoencoder, data, epochs=10000, tb_writer=tb_writer)


    imgs = create_recons(autoencoder, 10, latent_dims=LATENT_DIMS)
    plot_image_list(imgs)
    plt.show()
    tb_writer.close()

if __name__ == '__main__':
    main()
