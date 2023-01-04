<h1 align="center">
  <b>Texture VAE</b><br>
</h1>

Experiments generating textures using Variational Autoencoders (VAE).

[Live Demo](https://tlangmo.github.io/texture-vae/)

VAEs have the benefit of fast inference, making them suitable to be deployed in the browser or games.


### Texture Stiching
The VAE model can be used beyond simple sampling. Since the image generation is fully differentiateable, we can guide the image synthesis towards an objective.
See `notebooks/stich.ipynb` 

![Stiched Bricks](assets/brick_stich.gif?raw=true "Bricks Stiched")

### Basic Requirements
* Python >= 3.8
* PyTorch >= 1.10.2
* Linux OS
* CUDA compatible GPU

### Installation
```
# create venv and install dependencies
make setup

# download the training data
wget https://build-fitid-s3-bucket-download.s3.eu-central-1.amazonaws.com/motesque/bricks1000.zip
unzip bricks1000.zip

```

### Configuration
```
model:
  latent_dims: 32
  texture_size: 128
  kl_weight: 0.7
lr: 1e-4
epochs: 500
snapshots: "../snapshots"
train_images: "../bricks1000"
logs:
  logdir: "../tb/runs"
  experiment_name: "default"
```

### Usage
```
cd texture_vae
python train.py -c config.yml
```

Tensorboard
```
make tb
```
Http Dev Server with ONNX runtime
```
make http
```

## Training Notes
The training images have been compiled from high-res Creative Commons Brick Textures sourced at [Texture Ninja](https://www.texture.ninja)
Each high-res texture was randomly cropped into 128x128 images using `cli/crop_gen.py`
* 
* Latent code size does not have a dramatic effect. 32 seems to be enough
* The KLD weigthing has a huge influence on the sampling quality. While a lower value
  does improve reconstruction, the sampled images are of bad quality then.

### Reconstructed Bricks
![Reconstructed Bricks](assets/bricks_reconstructed.jpg?raw=true "Bricks Reconstructed")

### Sampled Bricks
![Sampled Bricks](assets/bricks_sampled.jpg?raw=true "Bricks Sampled")




## Learning Material
### VAE Introductions
* https://www.jeremyjordan.me/variational-autoencoders/
* https://avandekleut.github.io/vae/
* https://towardsdatascience.com/generating-images-with-autoencoders-77fd3a8dd368
* https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
### Interesting Papers
* [Learning to Generate Images With Perceptual Similarity Metrics](https://arxiv.org/pdf/1511.06409.pdf)
* [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://aclanthology.org/N19-1021.pdf)

### Code repos:
https://github.com/AntixK/PyTorch-VAE

