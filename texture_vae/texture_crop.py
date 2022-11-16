from PIL import Image
import glob
import pathlib
import math
from pathlib import Path
import random

def create_image_crops(root_dir:Path, num_crops: int, outdir:Path, crop_size: int):
    if not outdir.exists():
        outdir.mkdir(exist_ok=True)
    for idx, f in enumerate(root_dir.glob("*.jpg")):
        with Image.open(f.as_posix()) as im:
            im_1000 = im.resize((math.floor(1000*im.size[0]/im.size[1]),1000))
            for c_idx in range(num_crops):
                y_pos = random.randint(0,1000-crop_size)
                x_pos = random.randint(0,im_1000.size[0]-crop_size)
                box = (x_pos, y_pos, x_pos + crop_size, y_pos + crop_size)
                region = im_1000.crop(box)
                region.save(f"{outdir.as_posix()}/{idx}_{c_idx:04}.jpg")

if __name__ == "__main__":
    create_image_crops(Path("~/dev/bricks/").expanduser(), num_crops=10, outdir=Path("crops"), crop_size=512)