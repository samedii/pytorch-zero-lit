from PIL import Image
import numpy as np
import jax
from .vit_jax import models


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def test_lit():

    [name for name in models.model_configs.MODEL_CONFIGS if name.startswith("LiT")]

    model_name = "LiT-B16B"

    lit_model = models.get_model(model_name)
    # Loading the variables from cloud can take a while the first time...
    lit_variables = lit_model.load_variables()
    # Creating tokens from freeform text (see next section).
    tokenizer = lit_model.get_tokenizer()
    # Resizing images & converting value range to -1..1 (see next section).
    image_preprocessing = lit_model.get_image_preprocessing()
    # Preprocessing op for use in tfds pipeline (see last section).
    pp = lit_model.get_pp()

    # Note that this is a list of images with different shapes, not a four
    # dimensional tensor.
    # [image.shape for image in images_list]

    # images = image_preprocessing(images_list)
    # images.shape, images.min(), images.max()

    texts = [
        "itap of a cd player",
        "a photo of a truck",
        "gas station",
        "chainsaw",
        "a bad photo of colorful houses",
        "a photo of a cat",
    ]
    tokens = tokenizer(texts)
    tokens.shape

    # zimg, ztxt, out = lit_model.apply(lit_variables, images=images, tokens=tokens)

    _, ztxt, _ = lit_model.apply(lit_variables, tokens=tokens)
    ztxt.shape

    # JIT-compile image embedding function because there are lots of images.
    @jax.jit
    def embed_images(variables, images):
        zimg, _, _ = lit_model.apply(variables, images=images)
        return zimg

    images = image_preprocessing([np.array(Image.open("cat.png").convert("RGB"))])
    print(images.shape, images.min(), images.max())

    zimg = embed_images(lit_variables, images)

    # Compute similarities ...
    print(zimg.shape, ztxt.shape)
    sims = zimg @ ztxt.T
    print(sims.shape)
    print(sims)


if __name__ == "__main__":
    test_lit()
