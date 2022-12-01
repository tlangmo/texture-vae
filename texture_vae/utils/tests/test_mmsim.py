import pytest
import torch
import torchvision.io
from tqdm import tqdm

from texture_vae.utils.mmsim import MSSIM


@pytest.fixture()
def image_brick_0():
    img = torchvision.io.read_image("./assets/0_0000.jpg").float().div(255)
    return img.unsqueeze(0)


@pytest.fixture()
def image_brick_1():
    img = torchvision.io.read_image("./assets/0_0001.jpg").float().div(255)
    return img.unsqueeze(0)


def test_mmsim(image_brick_0, image_brick_1):
    sim = MSSIM()
    loss = sim(image_brick_0, image_brick_0)
    assert loss.item() == pytest.approx(0)

    loss = sim(image_brick_0, image_brick_0 + 0.01 * torch.randn_like(image_brick_0))
    assert loss.item() < 1e-2

    loss = sim(image_brick_0, image_brick_1)
    assert loss.item() < 0.9


import os

import torchvision.transforms.functional as F


def test_rebuild_from_noise(image_brick_0):
    noise_img = torch.randn_like(image_brick_0)
    params = torch.nn.Parameter(noise_img, requires_grad=True)
    opt = torch.optim.Adam(params=[params], lr=1e-1)
    EPOCHS = 200
    sim = MSSIM().to("cpu")
    # os.makedirs("./noise", exist_ok=True)
    for e in range(EPOCHS):
        opt.zero_grad()
        loss = sim(params, image_brick_0)
        assert not torch.all(torch.isnan(loss))
        loss.backward()
        opt.step()
        # if e % 10 == 0:
        #     pil_omg= F.to_pil_image(params[0])
        #     pil_omg.save(f"noise/{e:04}.jpg")
    assert loss < 0.5
