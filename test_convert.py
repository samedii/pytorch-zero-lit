# %%
from pathlib import Path
import json
from PIL import Image
import numpy as np
import jax
import tensorflow as tf
import torch
import torch.nn as nn
import onnx
import torch
from onnx2torch import convert
import jax.experimental.jax2tf
import jax.tree_util
import tf2onnx
import onnxruntime as rt

from lit.vit_jax import models

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

LARGE_MODELS = {"LiT-L16L", "LiT-L16S", "LiT-L16Ti"}


def convert_to_onnx(model_name):
    model_names = [
        name for name in models.model_configs.MODEL_CONFIGS if name.startswith("LiT")
    ]
    if model_name not in model_names:
        raise ValueError(f"Model name should be one of {model_names}")

    model = models.get_model(model_name)

    lit_variables = model.load_variables()
    tokenizer = model.get_tokenizer()

    image_preprocessing = model.get_image_preprocessing()

    texts = [
        "a photo of a cat",
    ]

    tokens = tokenizer(texts)
    _, ztxt, _ = model.apply(lit_variables, tokens=tokens)

    Path(f"tests/{model_name}").mkdir(exist_ok=True)

    Path(f"tests/{model_name}/tokens.json").write_text(json.dumps(tokens.tolist()))

    Path(f"tests/{model_name}/text_encoding.json").write_text(json.dumps(ztxt.tolist()))

    images = [np.array(Image.open("tests/cat.png").convert("RGB"))]
    preprocessed_image = image_preprocessing(images)

    Path(f"tests/{model_name}/preprocessed_image.json").write_text(
        json.dumps(preprocessed_image.tolist())
    )

    del tokenizer

    def encode_images(lit_variables, images):
        zimg, _, _ = model.apply(lit_variables, images=images)
        return zimg

    Path(f"tests/{model_name}/image_encoding.json").write_text(
        json.dumps(encode_images(lit_variables, preprocessed_image).tolist())
    )

    with_gradient = True
    compile_model = False
    enable_xla = False  # tf2onnx does not support a lot of the operations

    _, h, w, c = preprocessed_image.shape

    tf_fn = jax.experimental.jax2tf.convert(
        encode_images,
        polymorphic_shapes=[None, f"(b, {h}, {w}, {c})"],
        with_gradient=with_gradient,
        enable_xla=enable_xla,
    )
    print("Converted to tensorflow")

    param_vars = tf.nest.map_structure(
        lambda param: tf.Variable(param, trainable=with_gradient), lit_variables
    )

    tf_graph = tf.function(
        lambda inputs: tf_fn(param_vars, inputs),
        autograph=False,
        jit_compile=compile_model,
    )

    tf_graph.get_concrete_function(tf.TensorSpec([None, h, w, c], tf.float32))
    print("Converted to tensorflow graph")

    assert (
        np.abs(
            (
                tf_graph(preprocessed_image)
                - encode_images(lit_variables, preprocessed_image)
            ).numpy()
        ).max()
        <= 1e-3
    )

    # del model, lit_variables

    output_path = f"{model_name}.onnx"

    onnx_model, external_tensor_storage = tf2onnx.convert.from_function(
        tf_graph,
        input_signature=[tf.TensorSpec(shape=(None, h, w, c), dtype=tf.float32)],
        output_path=output_path,
        large_model=model_name in LARGE_MODELS,
    )
    return output_path


def convert_to_torch(onnx_path, model_name):
    torch_model = convert(onnx_path)
    torch.save(torch_model, f"{model_name}.pt")


if __name__ == "__main__":
    for model_name in LARGE_MODELS:
        torch_model = convert(f"{model_name}/__MODEL_PROTO.onnx")
        torch.save(torch_model, f"{model_name}.pt")

    # for model_name in ["LiT-B16B", "LiT-B16B_2", "LiT-L16L"]:
    #     # "LiT-L16S", "LiT-L16Ti", "LiT-L16S"
    #     print(model_name)
    #     if Path(f"{model_name}.pt").exists():
    #         continue

    #     onnx_path = Path(f"{model_name}.onnx")
    #     if not onnx_path.exists():
    #         onnx_path = convert_to_onnx(model_name)
    #         print("Converted to ONNX")
    #     convert_to_torch(onnx_path, model_name)
    #     print("Converted to pytorch")
