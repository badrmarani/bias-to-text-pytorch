import os
import warnings
from argparse import ArgumentParser

import pandas as pd
import torch
from torch import utils
from torch.serialization import SourceChangeWarning
from tqdm import tqdm

from commonalizer import seed_everything
from commonalizer.clip_prefix_captioning_inference import extract_caption
from commonalizer.dataset import CelebA
from commonalizer.keywords import extract_keywords
from commonalizer.metrics.clip import clip_score

warnings.filterwarnings("ignore", category=SourceChangeWarning)


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

    image_dir = "./data/celeba/img_align_celeba/"
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
                "correct",
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
                    abs_filename_path = os.path.join(image_dir, filenames[i])
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
        df = pd.read_csv(df_results_path, index_col=0)

    # extract keywords
    df_correct = df[df["correct"] == 0]
    df_wrong = df[df["correct"] == 1]

    # y: blond; pred: not blond
    df_wrong_class_0 = df_wrong[df_wrong["actual"] == 0]
    # y: not blond; pred: blond
    df_wrong_class_1 = df_wrong[df_wrong["actual"] == 1]

    # y: blond; pred: blond
    df_correct_class_0 = df_correct[df_correct["actual"] == 0]
    # y: blond; pred: blond
    df_correct_class_1 = df_correct[df_correct["actual"] == 1]

    caption_wrong_class_0 = " ".join(df_wrong_class_0["caption"].tolist())
    caption_wrong_class_1 = " ".join(df_wrong_class_1["caption"].tolist())

    keywords_wrong_class_0 = extract_keywords(caption_wrong_class_0)
    keywords_wrong_class_1 = extract_keywords(caption_wrong_class_1)

    # compute `clip score`
    score_wrong_class_0 = clip_score(
        image_dir, df_wrong_class_0["filename"], keywords_wrong_class_0
    )
    score_wrong_class_1 = clip_score(
        image_dir, df_wrong_class_1["filename"], keywords_wrong_class_1
    )

    score_correct_class_0 = clip_score(
        image_dir, df_correct_class_0["filename"], keywords_wrong_class_0
    )
    score_correct_class_1 = clip_score(
        image_dir, df_correct_class_1["filename"], keywords_wrong_class_1
    )

    cs_class_0 = score_wrong_class_0 - score_correct_class_0
    cs_class_1 = score_wrong_class_1 - score_correct_class_1
    print(cs_class_0, cs_class_1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="data/weights/best_model.pth"
    )
    parser.add_argument("--extract_captions", type=bool, default=True)

    args = parser.parse_args()
    main(**vars(args))
