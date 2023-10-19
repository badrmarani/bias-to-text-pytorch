import os

import clip
import skimage.io as io
import torch
from PIL import Image
from tqdm import trange

from .. import seed_everything


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
