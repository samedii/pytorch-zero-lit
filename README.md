# pytorch-zero-lit

Converted official JAX models for [LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/pdf/2111.07991v3.pdf)
to pytorch.

_JAX -> Tensorflow -> ONNX -> Pytorch._

- Image encoder is loaded into pytorch and supports gradients
- Text encoder is not loaded into pytorch and runs via ONNX on cpu

## Install

```bash
poetry add pytorch-zero-lit
```

or

```bash
pip install pytorch-zero-lit
```

## Usage

```python
from lit import LiT

model = LiT()

images = TF.to_tensor(
    Image.open("cat.png").convert("RGB").resize((224, 224))
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
```
