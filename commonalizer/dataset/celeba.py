import os

import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class CelebA(data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_name: str = "Blond_Hair",
        confounder_name: str = "Male",
    ) -> None:
        super(CelebA, self).__init__()

        self.root = root
        split_type = dict(train=0, valid=1, test=2)

        list_attr_celeba = pd.read_csv(
            os.path.join(root, "list_attr_celeba.csv"),
            delim_whitespace=True,
            header=1,
            index_col=0,
        )

        list_eval_partition = pd.read_csv(
            os.path.join(root, "list_eval_partition.csv"),
            delim_whitespace=True,
            index_col=0,
            header=None,
        )

        list_attr_celeba["partition"] = list_eval_partition
        list_attr_celeba = list_attr_celeba[
            list_attr_celeba["partition"] == split_type[split]
        ]

        # get the targets
        self.targets = list_attr_celeba[target_name].values
        self.targets[self.targets == -1] = 0
        self.targets_confounder = list_attr_celeba[confounder_name].values
        self.targets_confounder[self.targets_confounder == -1] = 0

        self.targets_groups = 2 * self.targets + self.targets_confounder
        self.targets_groups = self.targets_groups.astype("int")

        # get the filename and splits
        self.filenames = list_attr_celeba.index.tolist()

        # cvt to torch.Tensor
        self.targets = torch.from_numpy(self.targets)
        self.targets_confounder = torch.from_numpy(self.targets_confounder)
        self.targets_groups = torch.from_numpy(self.targets_groups)

    @property
    def transform(self):
        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_filename = os.path.join(self.root, "img_align_celeba", filename)
        input = Image.open(image_filename).convert("RGB")
        input = self.transform(input)
        target = self.targets[index]
        targets_confounder = self.targets_confounder[index]
        targets_groups = self.targets_groups[index]
        return input, target, targets_groups, targets_confounder, filename


if __name__ == "__main__":
    dataset = CelebA(root="data/celeba/", split="test")
    print(dataset)
