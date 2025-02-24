from llm_experiments import models
import base64
from io import BytesIO

from PIL import Image


def test_mxbai_embed_large():
    model = models.instantiate_embedding("mxbai-embed-large")
    assert model.embed_query("test")


def test_nomic_embed_text():
    model = models.instantiate_embedding("nomic-embed-text")
    assert model.embed_query("test")


def test_4o_mini():
    model = models.instantiate_chat("4o-mini")
    res = model.invoke("test").content
    print(res)
    assert res


def test_r1():
    model = models.instantiate_chat("deepseek")
    res = model.invoke("test")
    print(res)
    assert res


def test_llama():
    model = models.instantiate_chat("llama")
    res = model.invoke("test")
    print(res)
    assert res


def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def test_llava():
    pil_image = Image.open("tests/assets/public-domain-mickey-mouse.jpeg")
    image_b64 = convert_to_base64(pil_image)
    model = models.instantiate_vision("llava").bind(images=[image_b64])
    res = model.invoke("what is this image?")
    print(res)
    assert res
