import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize, ToTensor


class TextureDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = list(Path(self.root).glob("*.jpg"))
        self.files = list(sorted(self.files, key=TextureDataset.extract_label))

    @staticmethod
    def extract_label(fn):
        return int(fn.name.split("_")[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = read_image(img_path.as_posix())
        image = image.float().div(255)
        if self.transform:
            image = self.transform(image)
        return image, TextureDataset.extract_label(img_path)
