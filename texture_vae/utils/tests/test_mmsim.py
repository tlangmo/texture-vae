import pytest
import torch
from texture_vae.utils.mmsim import MSSIM
import torchvision.io
from tqdm import tqdm
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
    assert loss.item() <  1e-2

    loss = sim(image_brick_0,image_brick_1)
    assert loss.item() < 0.9

import time
def test_rebuild_from_noise(image_brick_0):
    noise_img =  torch.randn_like(image_brick_0)
    params = torch.nn.Parameter(noise_img, requires_grad=True)
    opt = torch.optim.Adam(params=[params], lr=1e-2)
    EPOCHS = 100
    for e in range(EPOCHS):

