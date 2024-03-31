import argparse
from typing import Dict, Any

import lightning as L
import torch
torch.set_float32_matmul_precision('high')

from model import Model
from data import C10kDataModule
from lightning.pytorch.loggers import WandbLogger


def pretraining(config:Dict[str, Any]):
    mnist_data = C10kDataModule(config)
    mnist_data.setup()
    if config["checkpoint_path"] is not None:
        model = Model.load_from_checkpoint(config["checkpoint_path"], config=config)
    else:
        model = Model(config)

    if config["no_log"]:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(project="C10k-better")

    trainer = L.Trainer(max_epochs=config["epochs"],
                        accelerator="gpu",
                        logger=wandb_logger,
                        log_every_n_steps=10,
                        precision="16-mixed",
                        benchmark=True,
    )

    trainer.fit(model, datamodule=mnist_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--width", type=int, default=1164)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", help="Disable logging (for debugging)")
    parser.add_argument("--backbone", type=str, default="mit-b2", help="backbone for the model")
    parser.add_argument("--augmentation_level", type=str, default="hard_v2", help="augmentation level for the data, check data.py")
    parser.add_argument("--TTA", action="store_true", help="whether to use test time augmentation")

    config = vars(parser.parse_args())
    pretraining(config)
