from torchvision.io import read_image
from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor, Resize
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
class TextureDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files = list(Path(self.img_dir).glob("*.jpg"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = read_image(img_path.as_posix())
        image = image.float().div(255)
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == "__main__":
    dl = TextureDataset("crops", Resize(64))
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3


    train_dataloader = DataLoader(dl, batch_size=18, shuffle=True)
    images = next(iter(train_dataloader))
    print(images)
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(dl), size=(1,)).item()
#     img = dl[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(img.permute(1,2,0), cmap="gray")
# plt.show()

