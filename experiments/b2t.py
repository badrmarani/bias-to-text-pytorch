import os
import warnings
from argparse import ArgumentParser

import pandas as pd
import torch
from torch import utils
from torch.serialization import SourceChangeWarning
from tqdm import tqdm

from b2t import seed_everything
from b2t.captions import extract_caption
from b2t.dataset import CelebA, Waterbirds
from b2t.keywords import extract_keywords
from b2t.metrics import clip_score

warnings.filterwarnings("ignore", category=SourceChangeWarning)


def clip_score_wrapper(images_dir, df_wrong, df_correct, keywords, **kwargs):
    if df_wrong.empty or df_correct.empty:
        score = torch.tensor([0.0])  # we can't decide
    else:
        score_wrong_class_0 = clip_score(
            images_dir,
            df_wrong["filename"],
            keywords,
            **kwargs,
        )
        score_correct_class_0 = clip_score(
            images_dir,
            df_correct["filename"],
            keywords,
            **kwargs,
        )
        score = score_wrong_class_0 - score_correct_class_0

    return score


@seed_everything(42)
@torch.no_grad()
def main(
    classification_model_path: str,
    dataset_name: str,
    captioning_model: str = None,
    extract_captions: bool = True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset_name == "celeba":
        root = "data/images/celeba/"
        val_dataset = CelebA(root=root, split="valid")
    elif dataset_name == "waterbirds":
        root = "data/images/waterbirds/"
        val_dataset = Waterbirds(root=root, split="valid")

    val_dataloader = utils.data.DataLoader(
        val_dataset,
        batch_size=2**8,  # 2**8 == 256
        num_workers=4,
        drop_last=False,
    )

    images_dir = val_dataset.root
    results_dir = os.path.join("data/results/b2t/", f"{dataset_name}/")
    os.makedirs(results_dir, exist_ok=True)

    f = os.path.join(results_dir, f"{captioning_model}_summary.csv")
    if not os.path.exists(f):
        model = torch.load(classification_model_path, map_location=device)
        model.eval()

        df = pd.DataFrame(
            columns=[
                "filename",
                "target",
                "prediction",
                "correct",
                "group",
                "confounder",
                f"caption/{captioning_model}",
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
                    abs_filename_path = os.path.join(images_dir, filenames[i])
                    caption = extract_caption(
                        abs_filename_path, captioning_model
                    )

                df.loc[len(df.index)] = {
                    "filename": filenames[i],
                    "target": targets[i].item(),
                    "prediction": predictions[i].item(),
                    "correct": correct[i].item(),
                    "group": targets_groups[i].item(),
                    "confounder": targets_confounder[i].item(),
                    f"caption/{captioning_model}": caption,
                }

                df.to_csv(f)

    else:
        df = pd.read_csv(f, index_col=0)

    # extract keywords

    df_wrong = df[df["correct"] == 0]
    # y: not blond; pred: blond
    df_wrong_class_0 = df_wrong[df_wrong["target"] == 0]
    # y: blond; pred: not blond
    df_wrong_class_1 = df_wrong[df_wrong["target"] == 1]

    df_correct = df[df["correct"] == 1]
    # y: not blond; pred: not blond
    df_correct_class_0 = df_correct[df_correct["target"] == 0]
    # y: blond; pred: blond
    df_correct_class_1 = df_correct[df_correct["target"] == 1]

    caption_class_0 = " ".join(
        df_wrong_class_0[f"caption/{captioning_model}"].tolist()
    )
    caption_class_1 = " ".join(
        df_wrong_class_1[f"caption/{captioning_model}"].tolist()
    )
    keywords_class_0 = extract_keywords(caption_class_0)
    keywords_class_1 = extract_keywords(caption_class_1)

    # compute `clip score`
    kwargs = dict(device=device, clip_weights_path="./data/pretrained_models/")

    score_class_0 = clip_score_wrapper(
        images_dir,
        df_wrong_class_0,
        df_correct_class_0,
        keywords_class_0,
        **kwargs,
    )
    score_class_1 = clip_score_wrapper(
        images_dir,
        df_wrong_class_1,
        df_correct_class_1,
        keywords_class_1,
        **kwargs,
    )

    # saving results
    df = pd.DataFrame(
        columns=[
            f"class_0/{captioning_model}_keyword",
            "class_0/clip_score",
            f"class_1/{captioning_model}_keyword",
            "class_1/clip_score",
        ],
    )

    if len(keywords_class_0):
        df["class_0/keyword"] = keywords_class_0
        df["class_0/score"] = score_class_0.cpu().numpy()
    if len(keywords_class_1):
        df["class_1/keyword"] = keywords_class_1
        df["class_1/score"] = score_class_1.cpu().numpy()

    df.to_csv(
        os.path.join(results_dir, f"{captioning_model}_summary_score.csv")
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--classification_model_path",
        type=str,
        default="data/pretrained_models/clf_resnet_erm_celeba.pth",
    )
    parser.add_argument(
        "--dataset_name", type=str, choices=["celeba", "waterbirds"]
    )
    parser.add_argument("--extract_captions", type=bool, default=True)
    parser.add_argument(
        "--captioning_model",
        type=str,
        choices=["clipcap", "git"],
        default="clipcap",
    )

    args = parser.parse_args()

    main(**vars(args))
