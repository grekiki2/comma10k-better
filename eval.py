import argparse
from typing import Dict, Any

import lightning as L
import torch
torch.set_float32_matmul_precision('high')

from model import Model
from data import C10kDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import StochasticWeightAveraging


def eval(config:Dict[str, Any]):
    mnist_data = C10kDataModule(config)
    mnist_data.setup()
    model = Model.load_from_checkpoint(config["checkpoint_path"], config=config)

    trainer = L.Trainer(max_epochs=1,
                        accelerator="gpu",
                        precision="16-mixed",
                        benchmark=True,
    )

    trainer.validate(model, datamodule=mnist_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--width", type=int, default=1164)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnet-b3", help="backbone for the model, for example resnet50")
    parser.add_argument("--augmentation_level", type=str, default="hard", help="augmentation level for the data, check data.py")
    parser.add_argument("--TTA", action="store_true", help="whether to use test time augmentation")

    config = vars(parser.parse_args())
    eval(config)
