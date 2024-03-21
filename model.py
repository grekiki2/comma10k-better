import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

import lightning as L

def acc_fn(out, mask):
    valid_mask_count = (mask != -100).sum()
    return (torch.argmax(out, dim=1) == mask).float().sum() / valid_mask_count

class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = smp.DeepLabV3Plus(
            encoder_name=config["backbone"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return {'loss': loss, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        if self.config["TTA"]:
            x_flipped = torch.flip(x, [3])
            out_flipped = self(x_flipped)
            out_flipped = torch.flip(out_flipped, [3])
            out = (out + out_flipped) / 2

        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
        return {'loss': loss, 'accuracy': accuracy}

    def configure_optimizers(self):
        print("Configuring optimizers")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.config['lr'],
                                                        # Steps per epoch is a bit harder to get since we need to wait for the dataloader to be created
                                                        steps_per_epoch = len(self.trainer.fit_loop._data_source.dataloader()),
                                                        epochs=self.config['epochs'],
                                                        pct_start=self.config['warmup_epochs']/self.config["epochs"])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }



