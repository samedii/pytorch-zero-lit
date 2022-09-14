# Pytorch-LiT

[LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/pdf/2111.07991v3.pdf)

Wrapping LiT implementation in JAX with [jax2torch](https://github.com/lucidrains/jax2torch) to allow
gradient backpropagation.

Installation of `JAX` and `tensorflow` is required at the moment. The
packages `flaxformer` and `t5x` are not on pypi and are copied into this
sourcecode but are only used relatively and will not conflict if another
version is installed.

## Install

```bash
poetry add pytorch-lit
```
