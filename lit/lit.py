from os import truncate
from typing import List
from pathlib import Path
from PIL import Image
import wget
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, logging
import onnxruntime
from lantern import Tensor

logging.set_verbosity_error()


MODEL_NAMES = {"LiT-B16B_2", "LiT-L16L"}
S3_BUCKET_URL = "https://s3.wasabisys.com/nextml-model-data/pytorch-lit"


class LiT(nn.Module):
    def __init__(self, model_name="LiT-B16B_2", cache_dir=Path("models")):
        """
        LiT model for image-text similarity. Image encoder is loaded from initialization
        into pytorch but the text encoder is loaded at inference and runs using onnx on cpu.

        Args:
            model_name (str): Name of the model to load. "LiT-B16B_2" or "LiT-L16L"
            cache_dir (str): Path to the directory where the model is cached.
        """
        super().__init__()

        if model_name not in MODEL_NAMES:
            raise ValueError(f"Model name should be one of {MODEL_NAMES}")

        self.model_name = model_name
        self.image_encoder_path = Path(cache_dir) / f"{model_name}-image-encoder.pt"
        self.text_encoder_path = Path(cache_dir) / f"{model_name}-text-encoder.onnx"

        if not self.image_encoder_path.exists():
            print(f"Downloading {self.image_encoder_path.stem}...")
            self.image_encoder_path.parent.mkdir(exist_ok=True)
            wget.download(
                f"{S3_BUCKET_URL}/{self.image_encoder_path.name}",
                str(self.image_encoder_path),
            )

        if not self.text_encoder_path.exists():
            print(f"Downloading {self.text_encoder_path.stem}...")
            self.text_encoder_path.parent.mkdir(exist_ok=True)
            wget.download(
                f"{S3_BUCKET_URL}/{self.text_encoder_path.name}",
                str(self.text_encoder_path),
            )

        self.image_size = (224, 224)
        self.image_encoder = (
            torch.load(self.image_encoder_path).eval().requires_grad_(False)
        )

        # verified tokenizer has the same vocab
        pretrained_name = "bert-base-uncased"
        self.n_text_tokens = 16
        self.text_tokenizer = BertTokenizer.from_pretrained(
            pretrained_name,
            sep_token="[PAD]",
            padding=True,
            truncation=True,
            max_length=self.n_text_tokens,
        )

        self.text_encoder = BertModel.from_pretrained(pretrained_name)

    def tokenize_texts(self, texts: List[str]):
        tokens = self.text_tokenizer(
            texts,
            return_tensors="pt",
            # sep_token="[MASK]",
            # mask_token=0,
            # padding=True,
            # truncation=True,
            # max_length=self.n_text_tokens,
        ).input_ids
        return F.pad(tokens, (0, self.n_text_tokens - tokens.shape[1]))

    def encode_texts(self, texts: List[str]):
        """
        Encode texts of max 16 tokens to latent space.

        Loads onnx model from disk and runs inference on cpu. Gradients
        are not supported.

        Args:
            texts (List[str]): Batch of texts.
        """
        tokens = self.tokenize_texts(texts)
        providers = ["CPUExecutionProvider"]
        m = onnxruntime.InferenceSession(
            str(self.text_encoder_path), providers=providers
        )
        output_names = ["Identity_1:0"]
        text_encodings = m.run(output_names, dict(inputs=tokens.numpy()))
        text_encodings = torch.from_numpy(np.array(text_encodings))[0]
        return text_encodings / text_encodings.norm(dim=1, keepdim=True)

    def encode_images(self, images: Tensor.dims("NCHW")) -> Tensor.dims("NK"):
        """
        Encode images of size (224, 224) and range [0, 1] to latent space.

        Args:
            images (Tensor.dims("NCHW")): Batch of images. Expected to be in the range [0, 1].
        """
        if images.shape[-2:] != self.image_size:
            raise ValueError(
                f"Expected images to be of size {self.image_size} but got {images.shape[-2:]}"
            )

        image_encodings = self.image_encoder(images.mul(2).sub(1).permute(0, 2, 3, 1))
        return image_encodings / image_encodings.norm(dim=1, keepdim=True)

    @staticmethod
    def cosine_similarity(encodings_a, encodings_b):
        return encodings_a @ encodings_b.T


def test_text_tokens():
    import json
    from pathlib import Path

    for model_name in MODEL_NAMES:
        model = LiT(model_name)

        texts = [
            "a photo of a cat",
        ]

        text_tokens = model.tokenize_texts(texts)
        reference_text_tokens = np.array(
            json.loads(Path(f"tests/{model_name}/text_tokens.json").read_text())
        )

        for reference, new in zip(reference_text_tokens[0], text_tokens[0]):
            assert reference == new


def test_text_encoding():
    import json
    from pathlib import Path

    torch.set_grad_enabled(False)

    for model_name in MODEL_NAMES:
        model = LiT(model_name)

        texts = [
            "a photo of a cat",
        ]

        text_encodings = model.encode_texts(texts)
        reference_text_encoding = np.array(
            json.loads(Path(f"tests/{model_name}/text_encoding.json").read_text())
        )
        assert np.abs(reference_text_encoding - text_encodings.numpy()).max() <= 1e-6


def test_image_encoding():
    import json
    from pathlib import Path
    import torchvision.transforms.functional as TF

    torch.set_grad_enabled(False)

    for model_name in MODEL_NAMES:
        model = LiT(model_name)

        images = TF.to_tensor(
            Image.open("tests/cat.png")
            .convert("RGB")
            .resize((224, 224), resample=Image.Resampling.BILINEAR)
        )[None]
        image_encodings = model.encode_images(images)

        reference_image_encoding = np.array(
            json.loads(Path(f"tests/{model_name}/image_encoding.json").read_text())
        )
        assert (
            np.abs(reference_image_encoding - np.array(image_encodings)).max() <= 2e-2
        )


def test_backpropagation():
    import torchvision.transforms.functional as TF

    torch.set_grad_enabled(False)

    for model_name in MODEL_NAMES:
        model = LiT(model_name)

        images = TF.to_tensor(
            Image.open("tests/cat.png")
            .convert("RGB")
            .resize((224, 224), resample=Image.Resampling.BILINEAR)
        )[None]
        images.requires_grad_()
        with torch.enable_grad():
            image_encodings = model.encode_images(images)
            image_encodings.mean().backward()
        assert images.grad is not None


def test_documentation_usage():
    from lit import LiT
    import torchvision.transforms.functional as TF

    model = LiT()

    images = TF.to_tensor(
        Image.open("tests/cat.png").convert("RGB").resize((224, 224))
    )[None]
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a fish",
    ]

    image_encodings = model.encode_images(images)
    text_encodings = model.encode_texts(texts)

    cosine_similarity = model.cosine_similarity(image_encodings, text_encodings)
    assert cosine_similarity[0].argmax() == 0
