import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

import lightning as L

def conv(ni, nf, ks=3, stride=2, act=True, bn=True):
    res = [nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2)]
    if act: res += [nn.ReLU()]
    if bn:  res += [nn.BatchNorm2d(nf, affine=True)]
    return res[0] if len(res) == 1 else nn.Sequential(*res)

def _conv_block(ni, nf, ks=3, stride=1, act=True, bn=True):
    return nn.Sequential(
        conv(ni, nf, ks=ks, stride=1, act=act, bn=bn),
        conv(nf, nf, ks=ks, stride=stride, act=None, bn=bn)
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=2):
        super().__init__()
        self.convs = _conv_block(ni, nf, ks=ks, stride=stride)
        self.idconv = nn.Identity() if ni==nf else conv(ni, nf, ks=1, stride=1, act=False)
        self.pool = nn.Identity() if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        
    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))
    
    def __iter__(self):
        yield self.convs
        yield self.idconv
        yield self.pool
        
    
def acc_fn(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()

class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = smp.Unet(
            encoder_name="resnet18",
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
        self.log('train_accuracy', accuracy, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return {'loss': loss, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        return {'loss': loss, 'accuracy': accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.config['lr'],
                                                        steps_per_epoch = len(self.trainer.fit_loop._data_source.dataloader()),
                                                        epochs=self.config['epochs'],
                                                        pct_start=self.config['pct_start'],)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }



