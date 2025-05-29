import argparse
import os
import warnings

import pytorch_lightning as pl
import torch

import b2t
from b2t.models import LightningBase, ResNet

warnings.filterwarnings("ignore")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_dir = args.images_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epochs = args.num_epochs
    num_classes = args.num_classes
    num_blocks = args.num_blocks
    dataset_name = args.dataset_name

    experiment_name = f"resnet{num_blocks}_" + os.path.basename(images_dir)

    if os.path.exists(f"checkpoints/{experiment_name}/"):
        print(f"Experiment {experiment_name} already exists, resuming training")
        ckpt = torch.load(f"checkpoints/{experiment_name}/last.ckpt")
    else:
        print(f"Starting new experiment {experiment_name}")
        ckpt = None

    # dataset

    dataset = getattr(b2t.datasets, dataset_name)

    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=dataset(root=images_dir, split="train"),
        val_dataset=dataset(root=images_dir, split="valid"),
        test_dataset=dataset(root=images_dir, split="test"),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # model
    model = ResNet(num_classes=num_classes, num_blocks=num_blocks)
    print(model)

    lightning_model = LightningBase(model).to(device=device)

    # callbacks

    progress_bar = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)

    logger = pl.loggers.TensorBoardLogger(save_dir=f"logs/{experiment_name}/")

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}/",
        filename="checkpoint-{epoch:03d}-{valid_bce:.5f}-{train_bce:.5f}",
        monitor="valid_bce",
        save_last=True,
        save_top_k=3,
        mode="min",
    )

    # training
    trainer = pl.Trainer(
        accelerator=device.type,
        benchmark=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
        callbacks=[ckpt_callback, progress_bar],
        logger=logger,
    )

    trainer.fit(lightning_model, datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="CelebA",
        choices=dir(b2t.datasets),
    )
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=18)
    args = parser.parse_args()

    train(args)
