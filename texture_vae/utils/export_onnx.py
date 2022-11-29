import torch.jit
import torch
from texture_vae.models.texture_vae import Autoencoder
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import os


def add_meta_data(model, md:dict):
    for k, v in md.items():
        meta = model.metadata_props.add()
        meta.key = k
        meta.value = f"{v}"
    return model

def export_to_onnx(snapshot: str, latent_dims: int , image_size:int, outfile:str):
    autoencoder = Autoencoder(latent_dims=latent_dims, image_size=image_size, device="cpu")
    try:
        autoencoder.load_state_dict(torch.load(snapshot))
    except Exception as err:
        raise ValueError("Cannot load weights file")

    # create scripted verion of torch module
    dummy_input = torch.distributions.Normal(0,1).sample((16, latent_dims)).to(autoencoder.device)
    scripted_module = torch.jit.script(autoencoder.decoder, example_inputs=[dummy_input.shape])
    torch.onnx.export(scripted_module, dummy_input, outfile, export_params=True,
                      verbose=True, dynamic_axes={'input' : {0 : 'batch_size'}},
                input_names=["input"], output_names=["output"])

    # Check that the model is well formed
    model = onnx.load(outfile)
    onnx.checker.check_model(model)
    add_meta_data(model, {"latent_dims": latent_dims, "image_size": image_size})
    onnx.save(model,outfile)


def validate_onnx(model_fn:str, outdir:str):
    ort_session = ort.InferenceSession(model_fn)
    md = ort_session.get_modelmeta().custom_metadata_map
    outputs = ort_session.run(None,
        {"input": 1*np.random.randn(10, int(md['latent_dims'])).astype(np.float32)},
    )
    images = outputs[0]

    def _save_image(idx: int, img_data, outdir:str):
        idx += 1
        try:
            im = Image.fromarray(img_data)
            im.save(f"{outdir}/onnx_sample_{idx:04}.jpg")
        except Exception as err:
            print(err)

    os.makedirs(outdir, exist_ok=True)
    images_uint8 = (images.reshape(-1, 3, images.shape[1], images.shape[1]).transpose(0, 2, 3,1) * 255).astype('uint8')
    for idx, img in enumerate([images_uint8[i] for i in range(images_uint8.shape[0])]):
        _save_image(idx, img, outdir)

if __name__ == "__main__":
    snaphot_fn = ""
    export_to_onnx(snaphot_fn,
               latent_dims=64, image_size=128, outfile="snapshot_lat64_res128_0.75_1000.onnx")


