import os

import clip
import skimage.io as io
import torch
from PIL import Image
from tqdm import tqdm

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

    batchs = [images[i : i + 2000] for i in range(0, len(images), 2000)]
    for batch in tqdm(batchs):
        # prepare the inputs
        image_inputs = torch.cat(
            [preprocess(pil_image).unsqueeze(0) for pil_image in batch]
        ).to(
            device
        )  # (1909, 3, 224, 224)

        text_inputs = torch.cat(
            [clip.tokenize(f"a photo of a {c}") for c in keywords]
        ).to(device)

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


if __name__ == "__main__":
    import pandas as pd

    clip_weights_path = "./data/weights/"
    image_dir = "./data/celeba/img_align_celeba/"

    df = pd.DataFrame(columns=["filename", "keyword"])
    df[len(df.index)] = dict(filename="174909.jpg", keywords="bla bla")

    device = torch.device("cpu")

    x = clip_score(
        image_dir, df["filename"], df["keyword"], device, clip_weights_path
    )

    print(x)
