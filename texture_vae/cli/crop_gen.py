import math
import random
from pathlib import Path

import click
from PIL import Image
from tqdm import tqdm


@click.command()
@click.argument("root_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--num_crops", type=int, help="Number of crops to generate for each image"
)
@click.option("--crop_size", type=int, default=128, help="Crop size in pixels")
@click.option(
    "--out_dir",
    type=click.Path(exists=False, path_type=Path),
    default=Path("./crops"),
    help="Output path",
)
def create_image_crops(root_dir: Path, num_crops: int, out_dir: Path, crop_size: int):
    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)
    files = list(root_dir.glob("*.jpg"))
    with tqdm(enumerate(files), unit="image", total=len(files)) as tq:
        for idx, f in tq:
            tq.set_description(f"Generating crops for {f.as_posix()}")
            with Image.open(f.as_posix()) as im:
                resized_to = int(256 * 1.5)
                im_1000 = im.resize(
                    (math.floor(resized_to * im.size[0] / im.size[1]), resized_to)
                )
                for c_idx in range(num_crops):
                    y_pos = random.randint(0, resized_to - crop_size)
                    x_pos = random.randint(0, im_1000.size[0] - crop_size)
                    box = (x_pos, y_pos, x_pos + crop_size, y_pos + crop_size)
                    region = im_1000.crop(box)
                    region.save(f"{out_dir.as_posix()}/{idx}_{c_idx:04}.jpg")


if __name__ == "__main__":
    create_image_crops()
