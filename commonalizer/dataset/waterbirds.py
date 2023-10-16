import os

import pandas as pd
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms


class Waterbirds(data.Dataset):
    def __init__(self, root: str, split: str = "train") -> None:
        super(Waterbirds, self).__init__()

        split = dict(train=0, valid=1, test=2).get(split, 0)

        self.root = os.path.join(root, "waterbird_complete95_forest2water2/")
        metadata_path = os.path.join(self.root, "metadata.csv")

        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata["split"] == split]

        # get the targets
        self.targets = self.metadata["y"].values
        self.targets[self.targets == -1] = 0
        self.targets_confounder = self.metadata["place"].values
        self.targets_confounder[self.targets_confounder == -1] = 0

        self.targets_groups = 2 * self.targets + self.targets_confounder
        self.targets_groups = self.targets_groups.astype("int")

        # Extract filenames and splits
        self.filenames = self.metadata["img_filename"].values

        # cvt to torch.Tensor
        self.targets = torch.from_numpy(self.targets)
        self.targets_confounder = torch.from_numpy(self.targets_confounder)
        self.targets_groups = torch.from_numpy(self.targets_groups)

    @property
    def transform(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_filename = os.path.join(self.root, filename)
        input = Image.open(image_filename).convert("RGB")
        input = self.transform(input)
        target = self.targets[index]
        targets_confounder = self.targets_confounder[index]
        targets_groups = self.targets_groups[index]
        return input, target, targets_groups, targets_confounder, filename


if __name__ == "__main__":
    dataset = Waterbirds("data/images/waterbirds/")
    input, target, targets_groups, targets_confounder, filename = dataset[20]
    print(
        input.size(),
        target.size(),
        targets_groups.size(),
        targets_confounder.size(),
    )
    print(filename)

    complete_path = os.path.join(dataset.root, filename)
    print(complete_path)
