import os

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
