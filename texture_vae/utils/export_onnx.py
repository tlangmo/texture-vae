import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.jit
from PIL import Image

from texture_vae.models.texture_vae import Autoencoder


def add_meta_data(model, md: dict):
    for k, v in md.items():
        meta = model.metadata_props.add()
        meta.key = k
        meta.value = f"{v}"
    return model


def export_to_onnx(snapshot: str, latent_dims: int, image_size: int, outfile: str):
    autoencoder = Autoencoder(
        latent_dims=latent_dims, image_size=image_size, device="cpu"
    )
    try:
        autoencoder.load_state_dict(torch.load(snapshot))
    except Exception as err:
        raise ValueError("Cannot load weights file", err)

    # create scripted verion of torch module
    dummy_input = (
        torch.distributions.Normal(0, 1)
        .sample((16, latent_dims))
        .to(autoencoder.device)
    )
    scripted_module = torch.jit.script(
        autoencoder.decoder, example_inputs=[dummy_input.shape]
    )
    torch.onnx.export(
        scripted_module,
        dummy_input,
        outfile,
        export_params=True,
        verbose=True,
        dynamic_axes={"input": {0: "batch_size"}},
        input_names=["input"],
        output_names=["output"],
    )

    # Check that the model is well formed
    model = onnx.load(outfile)
    onnx.checker.check_model(model)
    add_meta_data(model, {"latent_dims": latent_dims, "image_size": image_size})
    onnx.save(model, outfile)


def validate_onnx(model_fn: str, outdir: str):
    ort_session = ort.InferenceSession(model_fn)
    a = ort_session.get_inputs()
    md = ort_session.get_modelmeta().custom_metadata_map
    outputs = ort_session.run(
        None,
        {"input": 1 * np.random.randn(10, int(md["latent_dims"])).astype(np.float32)},
    )
    images = outputs[0]

    def _save_image(idx: int, img_data, outdir: str):
        idx += 1
        try:
            im = Image.fromarray(img_data)
            im.save(f"{outdir}/onnx_sample_{idx:04}.jpg")
        except Exception as err:
            print(err)

    os.makedirs(outdir, exist_ok=True)
    images_uint8 = (
        images.reshape(-1, 3, images.shape[1], images.shape[1]).transpose(0, 2, 3, 1)
        * 255
    ).astype("uint8")
    for idx, img in enumerate([images_uint8[i] for i in range(images_uint8.shape[0])]):
        _save_image(idx, img, outdir)

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("snapshot", type=str, help="Snapshot file to convert")
    parser.add_argument(
        "--latent-dims",
        "-l",
        type=int,
        required=True,
        help="Number of Latent Dimensions used in the snapshot",
    )
    parser.add_argument(
        "--image_size",
        "-r",
        type=int,
        required=True,
        help="Texture resolution used in the snapshot",
    )
    parser.add_argument(
        "--outfile", "-o", type=str, default=None, help="ONNX output file"
    )
    args = parser.parse_args()
    outfile = (
        f"{os.path.splitext(args.snapshot)[0]}.onnx"
        if args.outfile is None
        else args.outfile
    )
    export_to_onnx(
        args.snapshot,
        latent_dims=args.latent_dims,
        image_size=args.image_size,
        outfile=outfile,
    )


if __name__ == "__main__":
    main()
