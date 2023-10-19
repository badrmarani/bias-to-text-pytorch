import skimage.io as io
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .clip_prefix_captioning_inference import extract_caption_clipcap


@torch.no_grad()
def extract_caption_git(image_path: str):
    image = io.imread(image_path)
    pil_image = Image.fromarray(image)

    processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return generated_caption


def extract_caption(image_path: str, captioning_model: str = "clipcap"):
    assert captioning_model in ["clipcap", "git"]
    if captioning_model == "clipcap":
        caption = extract_caption_clipcap(image_path)
    elif captioning_model == "git":
        caption = extract_caption_git(image_path)
    return caption
