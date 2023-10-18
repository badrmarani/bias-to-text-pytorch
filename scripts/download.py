import os
from argparse import ArgumentParser

from torchvision.datasets import utils

os.environ["CURL_CA_BUNDLE"] = ""

# file id, filename
files = {
    "results/b2t_blond": [
        ("15kSeADU-yyIIqAMh8T40P1AYRTWpISrc", "b2t_celeba_blond_male.csv"),
    ],
    "pretrained_models": [
        ("1Ue8knFLZyePu36U22z4M7bB1eh-tVpX4", "clf_resnet_erm_celeba.pth"),
        ("1Ue8knFLZyePu36U22z4M7bB1eh-tVpX4", "clf_resnet_erm_waterbirds.pth"),
        ("1XsVzNBo_jW_ZTDN4rrgLia0zTWaEaPh6", "clipcap_coco_weights.pth"),
        ("1qQkq3zpgONNUogyi8GGIWLGQ8r_mnuNj", "clipcap_conceptual_weights.pth"),
    ],
    "images/waterbirds": [
        ("14k6fRRCKxCTqJ4HWe0EvwUtBEHclW_WT", "waterbirds.tar"),
    ],
    "images/celeba": [
        ("1kDqtHZHpYMe7rt1zu9pOevzUbApkDNRa", "list_eval_partition.csv"),
        ("1s8CyrddcxHdvwro-_M25H7uxsDWL_1Bs", "list_attr_celeba.csv"),
        ("1mGM-w9373aW5UJ27xa5oAsesL06JOe3h", "img_align_celeba.zip"),
    ],
}


def _check_integrity(root, type) -> bool:
    for _, filename in files.get(type):
        fpath = os.path.join(root, type, filename)
        _, ext = os.path.splitext(filename)
        # Allow original archive to be deleted (zip and 7z)
        # Only need the extracted images
        if ext not in [".zip", ".7z"] and not utils.check_integrity(fpath):
            return False


def download(root, type) -> None:
    if _check_integrity(root, type):
        print("Files already downloaded and verified")
        return

    for file_id, filename in files.get(type):
        utils.download_file_from_google_drive(
            file_id, os.path.join(root, type), filename
        )

    if type == "images/celeba":
        f = os.path.join(root, type, "img_align_celeba.zip")
        if os.path.exists(f):
            utils.extract_archive(f)
    elif type == "images/waterbirds":
        f = os.path.join(root, type, "waterbirds.tar")
        if os.path.exists(f):
            utils.extract_archive(f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--type", type=str, required=True, choices=files.keys())
    args = parser.parse_args()

    download(**vars(args))
