import os

import clip
import skimage.io as io
import torch
from PIL import Image
from tqdm import trange
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .. import seed_everything


def mean_pooling(model_output, attention_mask):
    # first element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


@seed_everything()
@torch.no_grad()
def git_score(root, filenames, keywords, device, git_weights_path):
    images = [os.path.join(root, f) for f in filenames]
    images = [Image.fromarray(io.imread(im)) for im in images]

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = AutoModel.from_pretrained("microsoft/git-base")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base")

    similarity_list = []

    encoded_input = tokenizer(
        [f"a photo of a {c}" for c in keywords],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    model_output = model(**encoded_input)
    text_inputs = mean_pooling(model_output, encoded_input["attention_mask"])

    num_steps = len(images) // 10 if len(images) > 10 else len(images)
    for i in trange(0, len(images), num_steps):
        batch = processor(
            images=images[i : i + num_steps], return_tensors="pt"
        ).pixel_values

        image_inputs = model.image_encoder(batch).last_hidden_state.mean(dim=1)

        image_features = image_inputs / image_inputs.norm(dim=-1, keepdim=True)
        text_features = text_inputs / text_inputs.norm(dim=-1, keepdim=True)
        similarity = 100.0 * image_features @ text_features.T
        similarity_list += [similarity]

    similarity = torch.cat(similarity_list).mean(dim=0)
    return similarity


@seed_everything()
@torch.no_grad()
def clip_score(root, filenames, keywords, device, clip_weights_path):
    images = [os.path.join(root, f) for f in filenames]
    images = [Image.fromarray(io.imread(im)) for im in images]

    clip_model, preprocess = clip.load(
        "ViT-B/32", device=device, download_root=clip_weights_path
    )

    similarity_list = []

    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in keywords]
    ).to(device=device)

    num_steps = len(images) // 10 if len(images) > 10 else len(images)
    for i in trange(0, len(images), num_steps):
        batch = images[i : i + num_steps]

        # prepare the inputs
        image_inputs = torch.cat(
            [preprocess(pil_image).unsqueeze(0) for pil_image in batch]
        ).to(device=device)

        # compute features
        image_features = clip_model.encode_image(image_inputs)
        text_features = clip_model.encode_text(text_inputs)

        # pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = 100.0 * image_features @ text_features.T  # (1909, 20)
        similarity_list += [similarity]

    similarity = torch.cat(similarity_list).mean(dim=0)

    return similarity
