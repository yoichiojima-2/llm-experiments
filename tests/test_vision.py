import base64
from io import BytesIO
from PIL import Image
from llm_experiments.vision import instantiate_vision


def convert_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def test_llava():
    pil_image = Image.open("tests/assets/public-domain-mickey-mouse.jpeg")
    image_b64 = convert_to_base64(pil_image)
    model = instantiate_vision("llava").bind(images=[image_b64])
    print(model.invoke("what is this image?"))