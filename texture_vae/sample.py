import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributions
from PIL import Image
from texture_vae.models.texture_vae import Autoencoder
import os
from texture_vae.utils.utils import create_variations


def sample():
    os.makedirs("../output128", exist_ok=True)
    LATENT_DIMS = 128
    TEXTURE_SIZE = 128
    autoencoder = Autoencoder(latent_dims=LATENT_DIMS, image_size=TEXTURE_SIZE, device="cpu")
    weights_fn = "/home/tlangmo/dev/texture-vae/snapshots/snapshot_lat128_res128_curated.pth"
    try:
        autoencoder.load_state_dict(torch.load(weights_fn))
        pass
    except FileNotFoundError as err:
        pass
    import time
    now = time.perf_counter()
    images = create_variations(autoencoder, 32, latent_dims=LATENT_DIMS,
                               image_size=TEXTURE_SIZE, channels=3)
    print(f"elpased ms = {(time.perf_counter()-now)*1000:.2f}")
    def save_image(idx: int, img_data):
        idx += 1
        im = Image.fromarray(img_data)
        im.save(f"../output128/brick_sample_{idx:04}.jpg")
    for idx, img in enumerate([images[i] for i in range(images.shape[0])]):
        save_image(idx, img)

if __name__ == '__main__':
    sample()