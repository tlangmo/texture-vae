from typing import List

import matplotlib.pyplot as plt
import torch


def create_recons(
    autoencoder, count, latent_dims, image_size: int, channels: int, img=None
):
    if img == None:
        # sample
        z = torch.distributions.Normal(0, 1).sample((count, latent_dims)).to("cuda")
        img = autoencoder.decoder(z)
    if channels == 3:
        imgs = (
            img.reshape(count, channels, image_size, image_size)
            .detach()
            .to("cpu")
            .permute(0, 2, 3, 1)
            .numpy()
            * 255
        ).astype("uint8")
    else:
        imgs = (
            img.reshape(count, channels, image_size, image_size)
            .detach()
            .to("cpu")
            .numpy()
            * 255
        ).astype("uint8")
    return imgs


def create_variations(
    autoencoder, count, latent_dims, image_size: int, channels: int, img=None
):
    z = (
        torch.distributions.Normal(0, 1)
        .sample((count, latent_dims))
        .to(autoencoder.device)
    )
    img = autoencoder.decoder(z)
    imgs = (
        img.reshape(count, channels, image_size, image_size)
        .detach()
        .to("cpu")
        .permute(0, 2, 3, 1)
        .numpy()
        * 255
    ).astype("uint8")
    return imgs


def plot_image_list(images: List):
    n_col = 5
    n_row = int(len(images) / n_col)
    # fig = plt.figure()
    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img.squeeze())
        ax.axis("off")
    return fig
