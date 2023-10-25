import pytorch_lightning as pl

from commonalizer.datasets import CelebA
from commonalizer.models import LightningBase, ResNet

datamodule = pl.LightningDataModule.from_datasets(
    train_dataset=CelebA(root="data/images/celeba", split="train"),
    valid_dataset=CelebA(root="data/images/celeba", split="valid"),
    test_dataset=CelebA(root="data/images/celeba", split="test"),
    batch_size=32,
    num_workers=0,
)

model = ResNet(num_classes=2)
model = LightningBase(model)

trainer = pl.Trainer()
trainer.fit(model, datamodule)
