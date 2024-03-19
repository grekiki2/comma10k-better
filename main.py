import lightning as L
import torch
torch.set_float32_matmul_precision('high')

from model import Model
from data import C10kDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from helpers import lsuv_init


def pretraining():
    mnist_data = C10kDataModule(batch_size=5)
    mnist_data.setup()
    config = {  'lr': 2e-4,
                'epochs': 20,
                'pct_start': 0.0, # we aim for ~3 epochs
            }
    model = Model(config)
    # lsuv_init(model.model, next(iter(mnist_data.train_dataloader()))[0][:16].cuda(), verbose=True)
    # model.model = torch.compile(model.model)

    wandb_logger = WandbLogger(project="C10k-better")
    # wandb_logger = None

    trainer = L.Trainer(max_epochs=config["epochs"],
                        accelerator="gpu",
                        logger=wandb_logger,
                        log_every_n_steps=10,
                        precision="16-mixed",
                        benchmark=True,
    )
    print(f"\n\n{'-'*20}\nStart training")
    trainer.fit(model, datamodule=mnist_data)


if __name__ == "__main__":
    pretraining()
