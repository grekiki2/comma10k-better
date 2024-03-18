import lightning as L
import torch
torch.set_float32_matmul_precision('high')

from model import Model
from data import C10kDataModule
from pytorch_lightning.loggers import WandbLogger
from helpers import lsuv_init


def pretraining():
    mnist_data = C10kDataModule(batch_size=32)
    mnist_data.setup()
    config = {  'lr': 6e-2,
                'epochs': 15,
                'pct_start': 0.2,}
    model = Model(config).cuda()
    lsuv_init(model.model, next(iter(mnist_data.train_dataloader()))[0], verbose=True)

    wandb_logger = WandbLogger(project="C10k-better")
    wandb_logger = None

    trainer = L.Trainer(max_epochs=config["epochs"],
                         accelerator="gpu",
                         logger=wandb_logger,
                         log_every_n_steps=1,
                         precision="bf16-mixed",
    )
    print(f"\n\n{'-'*20}\nStart training")
    trainer.fit(model, datamodule=mnist_data)




if __name__ == "__main__":
    pretraining()
