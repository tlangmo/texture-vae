import os
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import torch
import torch.distributions
import torch.utils.data
import torchvision
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm
from yaml import SafeLoader, load

import texture_vae.utils.texture_dataset as dataset
from texture_vae.models.texture_vae import Autoencoder
from texture_vae.utils.mmsim import MSSIM
from texture_vae.utils.utils import plot_image_list


def kl_loss(mu, log_var):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kld


from texture_vae.utils.mmsim import MSSIM


def train(
    autoencoder: Autoencoder,
    train_data,
    epochs: int,
    kl_weight: float,
    lr: float,
    post_epoch: Callable,
):
    sim = MSSIM()
    opt = torch.optim.Adam(params=autoencoder.parameters(), lr=lr)
    scheduler = CyclicLR(
        opt, base_lr=lr / 2, max_lr=lr * 2, step_size_up=100, cycle_momentum=False
    )
    for epoch in range(epochs):
        losses = []
        with tqdm(train_data, unit="batch") as tepoch:
            for x, l in tepoch:
                x = x.to(autoencoder.device)
                tepoch.set_description(f"epoch {epoch}")
                opt.zero_grad()
                x_hat, mu, logvar, z = autoencoder(x)
                # mse = torch.sum((x - x_hat) ** 2, dim=(1, 2, 3))
                # mse = mse.mean()
                sim_loss = sim(x, x_hat)
                kld = kl_loss(mu, logvar)
                kld = kld.mean()
                loss = (
                    sim_loss + kl_weight * kld
                )  # higher weight gives better generalization result in sampling
                loss.backward()
                opt.step()
                scheduler.step()
                losses.append(loss)
                tepoch.set_postfix(loss=loss.item())
        post_epoch(epoch, x_hat, sim_loss, kld, sum(losses) / len(losses))


def main(config: Dict):
    if not os.path.exists(config["snapshot"]):
        os.makedirs(config["snapshot"])

    LATENT_DIMS = config["model"]["latent_dims"]
    TEXTURE_SIZE = config["model"]["texture_size"]
    KL_WEIGHT = float(config["model"]["kl_weight"])
    LEARNING_RATE = float(config["lr"])
    SNAPSHOT_FN = Path(config["snapshot"]).as_posix()


    autoencoder = Autoencoder(
        latent_dims=LATENT_DIMS, image_size=TEXTURE_SIZE, device="cuda"
    )
    autoencoder.weight_init(0, 0.02)
    try:
        autoencoder.load_state_dict(torch.load(SNAPSHOT_FN))
    except FileNotFoundError as err:
        pass

    data = torch.utils.data.DataLoader(
        dataset.TextureDataset(
            config["train_images"],
            transform=torchvision.transforms.Resize(TEXTURE_SIZE),
        ),
        batch_size=25,
        shuffle=True,
    )
    summary(autoencoder, (3, TEXTURE_SIZE, TEXTURE_SIZE), device=autoencoder.device)
    input_images, labels = next(iter(data))
    images_plt = [
        (input_images[b].permute(1, 2, 0).numpy() * 255).astype("uint8")
        for b in range(input_images.shape[0])
    ]
    plot_image_list(images_plt)
    plt.show()
    plt.rcParams["figure.dpi"] = 300
    tb_writer = SummaryWriter(
        f"{config['logs']['logdir']}/{config['logs']['experiment_name']}"
    )

    def _log_epoch(epoch: int, x_hat, mse_loss, kl_loss, total_loss):
        torch.save(autoencoder.state_dict(), SNAPSHOT_FN)
        tb_writer.add_scalar("loss/mse", mse_loss.item(), epoch)
        tb_writer.add_scalar("loss/kl", kl_loss.item(), epoch)
        tb_writer.add_scalar("loss/total", total_loss.item(), epoch)
        tb_writer.add_images("reconstruction", x_hat, global_step=epoch)
        samples = torch.randn(16, LATENT_DIMS).to("cuda")
        sampled_images = autoencoder.sample(samples)
        tb_writer.add_images("sampled", sampled_images, global_step=epoch)

    train(
        autoencoder,
        data,
        epochs=config["epochs"],
        post_epoch=_log_epoch,
        kl_weight=KL_WEIGHT,
        lr=LEARNING_RATE,
    )
    tb_writer.close()


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option(
        "-c",
        "--config",
        dest="config_fn",
        default="config.yml",
        help="Config file",
        metavar="FILE",
    )
    (options, args) = parser.parse_args()
    with open(options.config_fn, "r") as fp:
        config = load(fp, SafeLoader)
    main(config)
