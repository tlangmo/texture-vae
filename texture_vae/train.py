import torch;# torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributions
import torchvision
import itertools
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
from texture_vae.models.texture_vae import Autoencoder
from typing import List
import os
from torch.utils.tensorboard import SummaryWriter
import texture_vae.utils.texture_dataset as dataset
from texture_vae.utils.utils import plot_image_list, create_recons
from  torch.optim.lr_scheduler import CyclicLR

#https://avandekleut.github.io/vae/
#https://openreview.net/attachment?id=ryguP1BFwr&name=original_pdf
#https://arxiv.org/abs/1511.06409 Learning to Generate Images With Perceptual Similarity Metrics
#https://torchmetrics.readthedocs.io/en/stable/image/multi_scale_structural_similarity.html
# channels not so important, but less reduce size a lot
# batchnorm is very important, don't do it without it
# intermediate fc layer before mu and log_var is not necessary
# reducing the learning rate  from 1e-4 to 1e-5 causes certainly worse results in the same 50 epochs, even 100 epochs.
# a lr schedule seems to be a good idea
# no tanh
# weight init is important
# the right tradeoff for kld is huge. Too much, and all is a blurry mush
# extra filter, extra conv layer - pretty much same loss result, but spped in training is better with more layers
# smaller lr does not help really, but cycling with inter epochs bounce works well base_lr=1e-4, max_lr=1e-3, step_size_up=150,
# latent code size makes a lot of differnece 16 -> 128
#https://github.com/AntixK/PyTorch-VAE

def kl_loss(mu, log_var):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kld

def train(autoencoder: Autoencoder,
          train_data,
          epochs: int,
          kl_weight: float,
          post_epoch: Callable):
    opt = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    scheduler = CyclicLR(opt, base_lr=1e-4, max_lr=1e-3, step_size_up=10, cycle_momentum=False)
    for epoch in range(epochs):
        losses = []
        latents = []
        labels = []
        with tqdm(train_data, unit="batch") as tepoch:
            for x, l in tepoch:
                x = x.to(autoencoder.device)
                tepoch.set_description(f"Epoch {epoch}")
                opt.zero_grad()
                x_hat, mu, logvar, z = autoencoder(x)
                mse = torch.sum((x-x_hat)**2)
                kld = kl_loss(mu, logvar)
                loss = mse + kl_weight*kld  # higher weight gives better generalization result in sampling
                loss.backward()
                opt.step()
                scheduler.step()
                losses.append(loss)
                latents.append(z)
                labels.append(l)
                tepoch.set_postfix(loss=loss.item())#, lr=scheduler.get_last_lr()[0])
        post_epoch(epoch, x_hat, mse, kld, sum(losses)/len(losses), latents, labels)
        tepoch.set_postfix(loss=sum(losses)/len(losses), lr=scheduler.get_last_lr()[0])

def main():
    if not os.path.exists("../snapshots"):
       os.makedirs("../snapshots")
    LATENT_DIMS = 128
    TEXTURE_SIZE = 128
    KL_WEIGHT = 2.5
    autoencoder = Autoencoder(latent_dims=LATENT_DIMS, image_size=TEXTURE_SIZE, device="cuda")
    autoencoder.weight_init(0, 0.02)
    SNAPSHOT = f"../snapshots/snapshot_lat{LATENT_DIMS}_res{TEXTURE_SIZE}_{KL_WEIGHT}_curated.pth"
    try:
        autoencoder.load_state_dict(torch.load(SNAPSHOT))
    except FileNotFoundError as err:
        pass
    data = torch.utils.data.DataLoader(dataset.TextureDataset('../crops50',
                                            transform=torchvision.transforms.Resize(TEXTURE_SIZE)),
                                       batch_size=25,
                                       shuffle=True)
    summary(autoencoder, (3, TEXTURE_SIZE, TEXTURE_SIZE), device=autoencoder.device)
    input_images, labels = next(iter(data))
    images_plt = [(input_images[b].permute(1, 2, 0).numpy() * 255).astype('uint8') for b in
                  range(input_images.shape[0])]
    plot_image_list(images_plt)
    plt.show()
    plt.rcParams['figure.dpi'] = 300
    tb_writer = SummaryWriter(f'../tensorboard/runs/texture-vae-lat{LATENT_DIMS}_res{TEXTURE_SIZE}_{KL_WEIGHT}_curated')

    def _log_epoch(epoch:int, x_hat, mse_loss, kl_loss, total_loss, latents, labels):
        torch.save(autoencoder.state_dict(), SNAPSHOT)
        epoch += 1000
        all_latents = torch.stack(latents, dim=0).reshape(-1, LATENT_DIMS)
        all_labels = torch.tensor(list(itertools.chain.from_iterable(labels)), dtype=torch.int32)
        if epoch % 10 == 0:
            torch.save({"latents":all_latents, "labels": all_labels }, f"./texture-vae-lat{LATENT_DIMS}_res{TEXTURE_SIZE}_{KL_WEIGHT}.pth")
        tb_writer.add_scalar('Loss/mse', mse_loss.item(), epoch)
        tb_writer.add_scalar('Loss/kl', kl_loss.item(), epoch)
        tb_writer.add_scalar('Loss/total', total_loss.item(), epoch)
        tb_writer.add_images("reconstruction", x_hat, global_step=epoch)
        samples = torch.randn(16,LATENT_DIMS).to("cuda")
        sampled_images = autoencoder.sample(samples)
        tb_writer.add_images("sampled", sampled_images, global_step=epoch)


    train(autoencoder, data, epochs=2000, post_epoch=_log_epoch, kl_weight=KL_WEIGHT)
    tb_writer.close()

def plot():
    from tsne_torch import TorchTSNE as TSNE
    plot_data= torch.load( f"./latents_720_curated.pth")
    lbl = plot_data["labels"]
    lbls = torch.where(lbl > 90, lbl, torch.zeros_like(lbl))
    lbls_mask = torch.nonzero(lbls).squeeze()
    lbls_used = lbls[lbls_mask].squeeze()
    samples_used = plot_data["latents"][lbls_mask]
    X_emb = TSNE(n_components=2, perplexity=30, n_iter=10000, verbose=True).fit_transform(samples_used)  # returns shape (n_samples, 2)
    np.save("emb.pth", X_emb)
    #X_emb = np.load("emb.pth.npy")
    plt.scatter(X_emb[:,0], X_emb[:,1], c=lbls_used.detach().cpu().numpy())
    plt.show()



if __name__ == '__main__':
    main()
    #plot()