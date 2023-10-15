import os
from argparse import ArgumentParser

from torchvision.datasets import utils

os.environ["CURL_CA_BUNDLE"] = ""


file_list = [
    # file id, md5 hash, filename
    ("1Ue8knFLZyePu36U22z4M7bB1eh-tVpX4", None, "best_model.pth"),
    ("1XsVzNBo_jW_ZTDN4rrgLia0zTWaEaPh6", None, "clipcap_coco_weights.pth"),
    (
        "1qQkq3zpgONNUogyi8GGIWLGQ8r_mnuNj",
        None,
        "clipcap_conceptual_weights.pth",
    ),
]

base_folder = "weights/"


def _check_integrity(root, file_list) -> bool:
    for _, md5, filename in file_list:
        fpath = os.path.join(root, base_folder, filename)
        _, ext = os.path.splitext(filename)
        # Allow original archive to be deleted (zip and 7z)
        # Only need the extracted images
        if ext not in [".zip", ".7z"] and not utils.check_integrity(fpath, md5):
            return False


def download(root) -> None:
    if _check_integrity(root, file_list):
        print("Files already downloaded and verified")
        return

    for file_id, md5, filename in file_list:
        utils.download_file_from_google_drive(
            file_id, os.path.join(root, base_folder), filename, md5
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/")
    args = parser.parse_args()

    download(**vars(args))
