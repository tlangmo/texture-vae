import pytest
from texture_vae.texture_vae_tng import VariationalEncoder, VariationalDecoder
from torchsummary import summary
import torch

def test_encoder_summary():
    ve = VariationalEncoder(img_size=64, latent_dims=512)
    summary(ve, (3, 64, 64), device='cpu')

def test_decoder_summary():
    vd = VariationalDecoder(latent_dims=512, encoded_img_size=8, encoded_channels=128)
    summary(vd, (512,), device='cpu')

@pytest.mark.skipif( not torch.cuda.is_available(), reason="cuda not available")
def test_encoder_summary_cuda():
    vd = VariationalDecoder(latent_dims=512, encoded_img_size=8, encoded_channels=128)
    vd = vd.to("cuda")
    summary(vd, (512,), device='cuda')