# Pytorch-LiT

[LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/pdf/2111.07991v3.pdf)

Wrapping LiT implementation in JAX with [jax2torch](https://github.com/lucidrains/jax2torch) to allow
gradient backpropagation.

Installation of `JAX` and `tensorflow` is required at the moment. The
packages `flaxformer` and `t5x` are not on pypi and are copied into this
sourcecode but are only used relatively and will not conflict if another
version is installed.

## Usage

```python
from lit import LiT


model = LiT()


image_encodings = model.encode_images()
text_encodings = model.encode_texts()

cosine_similarity = model.cosine_similarity(image_encodings, text_encodings)
```

## Install

```bash
poetry add pytorch-lit
```

or

```bash
pip install pytorch-lit
```

### CUDA Toolkit and cuDNN for tensorflow

Recommended to use miniconda to isolate CUDA and cuDNN and poetry for
python packages.

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Pytorch for CUDA 11.3

```bash
poetry run pip uninstall torch torchvision -y && poetry run pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### JAX for CUDA >= 11.1 and cuDNN >= 8.0.5

```bash
poetry run pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
