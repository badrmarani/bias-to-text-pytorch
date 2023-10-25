import gc

import torch
from torch import nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(ResNet, self).__init__()

        self.seq = models.resnet18(
            num_classes,
            weights="ResNet18_Weights.DEFAULT",
        )
        in_features = self.seq.fc.in_features

        del self.seq.fc
        gc.collect()

        self.seq.fc = nn.Linear(in_features, num_classes)

    def forward(self, inputs):
        return self.seq(inputs)


if __name__ == "__main__":
    model = ResNet()
    # print(model.seq)

    x = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    logits = model(x)
    print(logits.size())
