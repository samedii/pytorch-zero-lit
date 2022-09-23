from PIL import Image
import numpy as np
import jax
import tensorflow as tf
import torch
import torch.nn as nn
from jax2torch import jax2torch

from .vit_jax import models

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LiT(nn.Module):
    def __init__(self, model_name="LiT-B16B"):
        """
        ["LiT-B16B", "LiT-B16B_2", "LiT-L16L", "LiT-L16S", "LiT-L16Ti"]
        """
        super().__init__()

        model_names = [
            name
            for name in models.model_configs.MODEL_CONFIGS
            if name.startswith("LiT")
        ]
        if model_name not in model_names:
            raise ValueError(f"Model name should be one of {model_names}")

        self.model_name = model_name
        self.model = models.get_model(model_name)

        self.lit_variables = self.model.load_variables()
        self.tokenizer = self.model.get_tokenizer()

        self.image_preprocessing = self.model.get_image_preprocessing()

    # @jax2torch
    def encode_texts(self, texts):
        tokens = self.tokenizer(texts)
        _, ztxt, _ = self.model.apply(self.lit_variables, tokens=tokens)
        return ztxt

    # @jax2torch
    def encode_images(self, images):
        zimg, _, _ = self.model.apply(
            self.lit_variables, images=self.image_preprocessing(images)
        )
        return zimg

    @staticmethod
    def cosine_similarity(encodings_a, encodings_b):
        return encodings_a @ encodings_b.T


def test_lit():
    import json
    from pathlib import Path

    model = LiT()

    texts = [
        "itap of a cd player",
        "a photo of a truck",
        "gas station",
        "chainsaw",
        "a bad photo of colorful houses",
        "a photo of a cat",
    ]

    text_encodings = model.encode_texts(texts)

    images = [np.array(Image.open("tests/cat.png").convert("RGB"))]
    image_encodings = model.encode_images(images)

    cosine_similarity = model.cosine_similarity(image_encodings, text_encodings)

    reference_image_encoding = np.array(
        json.loads(Path("tests/cat_encoding.json").read_text())
    )
    assert np.abs(reference_image_encoding - np.array(image_encodings)).max() <= 1e-6
