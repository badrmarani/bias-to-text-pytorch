import os
import random
import warnings
from argparse import ArgumentParser
from contextlib import contextmanager

import pandas as pd
import torch
from torch import utils
from torch.serialization import SourceChangeWarning
from tqdm import tqdm

import numpy as np
from commonalizer.dataset import CelebA
from commonalizer.utils.clip_prefix_captioning_inference import extract_caption

warnings.filterwarnings("ignore", category=SourceChangeWarning)


@contextmanager
def seed_everything(seed=42):
    random.seed(seed)
    np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    yield


@seed_everything(42)
@torch.no_grad()
def main(model_path, extract_captions):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_dataset = CelebA(root="data/celeba/", split="valid")
    val_dataloader = utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=128,
        num_workers=4,
        drop_last=False,
    )

    results_dir = "results/b2t/celeba_blond_male/"
    df_results_path = os.path.join(results_dir, "b2t_celeba_blond_male.csv")

    if not os.path.exists(df_results_path):
        model = torch.load(model_path, map_location=device)
        model.eval()

        df = pd.DataFrame(
            columns=[
                "filename",
                "target",
                "prediction",
                "group",
                "confounder",
                "caption",
            ],
        )

        iterator = tqdm(val_dataloader)
        for index, batch in enumerate(iterator, start=1):
            (
                inputs,
                targets,
                targets_groups,
                targets_confounder,
                filenames,
            ) = batch
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            _, predictions = torch.max(logits, 1)

            correct = predictions == targets

            for i in range(predictions.size(0)):
                iterator.set_description(f"(batch-{index}) {filenames[i]}")

                caption = None
                if extract_captions:
                    abs_filename_path = os.path.join(
                        "./data/celeba/img_align_celeba/", filenames[i]
                    )
                    caption = extract_caption(abs_filename_path)

                df.loc[len(df.index)] = dict(
                    filename=filenames[i],
                    target=targets[i].item(),
                    prediction=predictions[i].item(),
                    correct=correct[i].item(),
                    group=targets_groups[i].item(),
                    confounder=targets_confounder[i].item(),
                    caption=caption,
                )

        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(df_results_path)

    else:
        df = pd.read_csv(df_results_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="data/weights/best_model.pth"
    )
    parser.add_argument("--extract_captions", type=bool, default=True)

    args = parser.parse_args()
    main(**vars(args))
