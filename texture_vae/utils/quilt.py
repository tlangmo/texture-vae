import onnxruntime as ort
import numpy as np
from PIL import Image
import os


def _save_image(idx: int, img_data, outdir: str):
    idx += 1
    try:
        im = Image.fromarray(img_data)
        im.save(f"{outdir}/onnx_sample_{idx:04}.jpg")
    except Exception as err:
        print(err)

def generate_patches(base_latent: np.ndarray, sigma:float, num_batches:int, batch_size:int, ort_session: ort.InferenceSession) -> np.ndarray:
    for b in range(num_batches):
        batch_input = np.repeat(base_latent, batch_size, axis=0) + np.random.normal(0, sigma, size=(batch_size, base_latent.shape[1])).astype(np.float32)
        outputs = ort_session.run(None,
                                  {"input": batch_input},
                                  )
        images = outputs[0]
        for i in range(images.shape[0]):
            images_uint8 = (images[i].reshape(3, 128, 128).transpose(1, 2, 0) * 255).astype('uint8')
            yield images_uint8, batch_input[i]

def generate_single_patch(base_latent: np.ndarray, ort_session: ort.InferenceSession) -> np.ndarray:
    batch_input = np.repeat(base_latent, 1, axis=0)
    outputs = ort_session.run(None,
                              {"input": batch_input},
                              )
    images = outputs[0]
    for i in range(images.shape[0]):
        images_uint8 = (images[i].reshape(3, 128, 128).transpose(1, 2, 0) * 255).astype('uint8')
        yield images_uint8, batch_input[i]

def debug_save_image(img_data, fn:str):
    try:
        im = Image.fromarray(img_data)
        im.save(fn)
    except Exception as err:
        print(err)


def l2_overlap_diff(patch, overlap, quilted_image, y, x):
    block_size = 128
    error = 0
    if x > 0:
        left = patch[:, :overlap] - quilted_image[y:y+block_size, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - quilted_image[y:y+overlap, x:x+block_size]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - quilted_image[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)
    return error

def find_best_patch(ort_session, base_latent, sigma, quilted_image,
                    dst_x,
                    dst_y,
                    overlap):
    e_min = (1e10,None)
    for candidate, lat in generate_patches(ort_session=ort_session,
                          base_latent=base_latent,
                          num_batches=10,
                          batch_size=16,
                          sigma=sigma):
        error = l2_overlap_diff(candidate, overlap, quilted_image, dst_y, dst_x)
        if error < e_min[0]:
            e_min = (error, lat)

    return list(generate_single_patch(np.atleast_2d(e_min[1]),ort_session))[0][0]






def quilt(width: int, height: int):
    ort_session = ort.InferenceSession("/home/tlangmo/dev/texture-vae/texture_vae/utils/snapshot_lat128_res128_curated.onnx")
    md = ort_session.get_modelmeta().custom_metadata_map
    image_size = int(md['image_size'])
    #base_latent = np.random.randn(1,int(md['latent_dims'])).astype(np.float32)
    base_latent = np.random.normal(0, 1, size=(1, int(md['latent_dims']))).astype(np.float32)
    #for idx, img in enumerate(generate_patches(base_latent, sigma=0.4, ort_session=ort_session, num_batches=1, batch_size=8)):
        #debug_save_image(img, fn=f"sample_{idx:04}.jpg")

    quilted_image = np.zeros((width,height,3), dtype=np.uint8)
    block_size = image_size
    overlap = block_size // 6
    num_block_w = width // block_size
    num_block_h = height // block_size
    print(num_block_w, num_block_h)
    for b_y in range(num_block_h):
        for b_x in range(num_block_w):
            x_pos = b_x * (image_size-overlap)
            y_pos = b_y * (image_size-overlap)
            quilted_image[x_pos:x_pos+image_size,y_pos:y_pos+image_size,:] = find_best_patch(ort_session=ort_session,
                                                                                             base_latent=base_latent,
                                                                                             sigma=1,
                                                                                             quilted_image=quilted_image,
                                                                                             dst_x=x_pos,
                                                                                             dst_y=y_pos,
                                                                                             overlap=overlap
                                                                                            )

    debug_save_image(quilted_image, fn=f"quilt.jpg")



quilt(1024,1024)
# def validate_onnx(model_fn:str, outdir:str):
#     ort_session = ort.InferenceSession(model_fn)
#     md = ort_session.get_modelmeta().custom_metadata_map
#     print(md)
#     outputs = ort_session.run(None,
#         {"input": 1*np.random.randn(10, int(md['latent_dims'])).astype(np.float32)},
#     )
#     images = outputs[0]