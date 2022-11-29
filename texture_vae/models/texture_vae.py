import torch;# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributions
import torchvision
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Tuple
from dataclasses import dataclass
from torchsummary import summary
from tqdm import tqdm
from typing import Callable

class VariationalEncoder(nn.Module):

    def __init__(self, latent_dims: int, img_size: int):
        super(VariationalEncoder, self).__init__()
        img_size = img_size
        channels = 16
        l0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(4,4), stride=(2,2), padding=1),
                                     torch.nn.BatchNorm2d(channels),
                                     torch.nn.LeakyReLU())
        img_size //= 2
        l1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels*2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            torch.nn.LeakyReLU(),
       #     torch.nn.BatchNorm2d(channels*2)
        )


        channels *= 2
        img_size //= 2

        l2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            torch.nn.LeakyReLU(),
        #    torch.nn.BatchNorm2d(channels * 2)
        )
        channels *= 2
        img_size //= 2

        l3 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1),
            torch.nn.LeakyReLU(),
       #     torch.nn.BatchNorm2d(channels * 2)
        )
        channels *= 2
        img_size //= 2

        self.conv_layers = nn.Sequential(l0, l1, l2, l3)
        self.fc_mu = nn.Linear(channels * img_size**2,  latent_dims)
        self.fc_log_variance = nn.Linear(self.fc_mu.in_features, self.fc_mu.out_features)
        self.encoded_channels = channels
        self.encoded_img_size = img_size

    def forward(self, x):
        bs = x.shape[0] # batch size
        x = self.conv_layers(x)
        x = x.view(bs, self.fc_mu.in_features)
        mu = self.fc_mu(x)
        log_var = self.fc_log_variance(x)
        return mu, log_var


class VariationalDecoder(nn.Module):
    def __init__(self,  latent_dims: int, encoded_channels: int, encoded_img_size:int):
        super(VariationalDecoder, self).__init__()
        self.encoded_channels = encoded_channels
        self.encoded_img_size = encoded_img_size
        self.fc = nn.Linear(latent_dims, encoded_channels * encoded_img_size**2)

        l0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=encoded_channels, out_channels=encoded_channels//2, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1,1)),
            torch.nn.LeakyReLU(),
          #  torch.nn.BatchNorm2d(encoded_channels//2)
        )
        encoded_channels //= 2

        l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=encoded_channels, out_channels=encoded_channels // 2, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1)),
            torch.nn.LeakyReLU(),
          #  torch.nn.BatchNorm2d(encoded_channels // 2)
        )
        encoded_channels //= 2


        l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=encoded_channels, out_channels=encoded_channels // 2, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1)),
            torch.nn.LeakyReLU(),
       #     torch.nn.BatchNorm2d(encoded_channels // 2)
        )
        encoded_channels //= 2

        l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=encoded_channels, out_channels=encoded_channels // 2, kernel_size=(4, 4),
                               stride=(2, 2), padding=(1, 1)),
            torch.nn.LeakyReLU(),
         #   torch.nn.BatchNorm2d(encoded_channels // 2)
        )
        encoded_channels //= 2

        cl_last =  nn.ConvTranspose2d(in_channels=encoded_channels, out_channels=3, kernel_size=(3, 3), stride=(1, 1),
                                 padding=(1,1))
        self.conv_layers = nn.Sequential(l0,l1,l2, l3, cl_last)

    def forward(self, z):
        z = self.fc(z)
        z = F.leaky_relu(z)
        z = z.reshape((-1,  self.encoded_channels, self.encoded_img_size, self.encoded_img_size))
        z = self.conv_layers(z)
        z = F.sigmoid(z)
        return z

#https://github.com/podgorskiy/VAE/blob/master/VAE.py
# #https://www.echevarria.io/blog/lego-face-vae/
# #https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/
class Autoencoder(nn.Module):
    def __init__(self, latent_dims, image_size:int, device:str = "cpu"):
        super(Autoencoder, self).__init__()
        self.device = device
        self.image_size = image_size
        self.latent_dims = latent_dims
        self.encoder = VariationalEncoder(latent_dims, image_size).to(device)
        self.decoder = VariationalDecoder(latent_dims, self.encoder.encoded_channels, self.encoder.encoded_img_size).to(device)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self,x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, z

    def weight_init(self, mean, std):
        for m in self.encoder._modules:
            self.normal_init(self.encoder._modules[m], mean, std)
        for m in self.decoder._modules:
            self.normal_init(self.decoder._modules[m], mean, std)

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
#
    def sample(self, latents: torch.Tensor):
        with torch.inference_mode():
            imgs = self.decoder(latents)
        return imgs

#
#
# def train(autoencoder: Autoencoder,
#           train_data,
#           epochs: int,
#           post_epoch: Callable):
#     opt = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
#     for epoch in range(epochs):
#         with tqdm(train_data, unit="batch") as tepoch:
#             for x in tepoch:
#                 if isinstance(x, list):
#                     x = x[0]
#                 x = x.to(autoencoder.device)
#                 tepoch.set_description(f"Epoch {epoch}")
#                 opt.zero_grad()
#                 x_hat, mu, logvar = autoencoder(x)
#                 kl = kl_loss(mu, logvar)
#                 mse = ((x-x_hat)**2).sum()
#                 loss = mse + 0.1 * kl
#                 loss.backward()
#                 opt.step()
#                 tepoch.set_postfix(loss=loss.item())
#         post_epoch(epoch, x_hat, mse, kl, loss)
#
#
# def create_recons(autoencoder, count, latent_dims, image_size:int, channels:int, img=None):
#     if img == None:
#         # sample
#         z = torch.distributions.Normal(0,1).sample((count, latent_dims)).to('cuda')
#         img = autoencoder.decoder(z)
#     if channels == 3:
#         imgs = (img.reshape(count, channels,image_size,image_size).detach().to('cpu').permute(0,2,3,1).numpy()*255).astype('uint8')
#     else:
#         imgs = (img.reshape(count, channels, image_size, image_size).detach().to('cpu').numpy() * 255).astype(
#             'uint8')
#     return imgs
#
# from typing import List
#
# def plot_image_list(images: List):
#     n_col = 5
#     n_row = int(len(images) / n_col)
#     #fig = plt.figure()
#     fig, axs = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
#     axs = axs.flatten()
#     for img, ax in zip(images, axs):
#         ax.imshow(img.squeeze())
#         ax.axis('off')
#     return fig
#
# #https://avandekleut.github.io/vae/
# import os
# from torch.utils.tensorboard import SummaryWriter
# import texture_dataset
#
#
# #https://openreview.net/attachment?id=ryguP1BFwr&name=original_pdf
#
# #https://arxiv.org/abs/1511.06409 Learning to Generate Images With Perceptual Similarity Metrics
# #https://torchmetrics.readthedocs.io/en/stable/image/multi_scale_structural_similarity.html
#
# # channels not so important, but less reduce size a lot
# # batchnorm is very important, don't do it without it
# # intermediate fc layer before mu and log_var is not necessary
# # reducing the learning rate  from 1e-4 to 1e-5 causes certainly worse results in the same 50 epochs, even 100 epochs.
# # a lr schedule seems to be a good idea
#
# def main():
#     if not os.path.exists("../snapshots"):
#        os.makedirs("../snapshots")
#     LATENT_DIMS = 512
#     TEXTURE_SIZE = 64
#     autoencoder = Autoencoder(latent_dims=LATENT_DIMS, image_size=TEXTURE_SIZE, device="cuda")
#     SNAPSHOT = f"../snapshots/snapshot_{LATENT_DIMS}_{TEXTURE_SIZE}.pth"
#     try:
#         autoencoder.load_state_dict(torch.load(SNAPSHOT))
#     except FileNotFoundError as err:
#         pass
#     data = torch.utils.data.DataLoader(texture_dataset.TextureDataset('./crops_many_more2',
#                                             transform=torchvision.transforms.Resize(TEXTURE_SIZE)),
#                                        batch_size=16,
#                                        shuffle=True)
#     summary(autoencoder, (3, TEXTURE_SIZE, TEXTURE_SIZE), device=autoencoder.device)
#
#     input_images = next(iter(data))
#     images_plt = [(input_images[b].permute(1, 2, 0).numpy() * 255).astype('uint8') for b in
#                   range(input_images.shape[0])]
#     plot_image_list(images_plt)
#     plt.show()
#     plt.rcParams['figure.dpi'] = 300
#     tb_writer = SummaryWriter(f'../tensorboard/runs/texture-vae-{LATENT_DIMS}_{TEXTURE_SIZE}_32c')
#
#     def _log_epoch(epoch:int, x_hat, mse_loss, kl_loss, total_loss):
#         torch.save(autoencoder.state_dict(), SNAPSHOT)
#         tb_writer.add_scalar('Loss/mse', mse_loss.item(), epoch)
#         tb_writer.add_scalar('Loss/kl', kl_loss.item(), epoch)
#         tb_writer.add_scalar('Loss/total', total_loss.item(), epoch)
#         images = create_recons(autoencoder, 16, latent_dims=LATENT_DIMS,
#                                image_size=TEXTURE_SIZE, channels=3, img=x_hat[:16])
#         fig_gen = plot_image_list(images)
#         tb_writer.add_figure("reconstruction", fig_gen, global_step=epoch)
#
#     train(autoencoder, data, epochs=50, post_epoch=_log_epoch)
#     tb_writer.close()
#
# if __name__ == '__main__':
#     main()