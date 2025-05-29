import pytorch_lightning as pl
import torch
from torch import nn

# flake8: noqa
from .resnet import ResNet


class LightningBase(pl.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets, _, _, _ = batch
        logits = self(inputs)

        loss = self.loss_fn(logits, targets)
        self.log(
            "train",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _, _, _ = batch
        logits = self(inputs)

        loss = self.loss_fn(logits, targets)
        self.log(
            "valid",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
