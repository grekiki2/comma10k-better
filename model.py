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
        conv(nf, nf, ks=ks, stride=stride, act=None, bn=False)
    )

class ResBlock(nn.Module):
    def __init__(self, ni, nf, ks=3, stride=2, bn=True):
        super().__init__()
        self.convs = _conv_block(ni, nf, ks=ks, stride=stride, bn=bn)
        self.idconv = nn.Identity() if ni==nf else conv(ni, nf, ks=1, stride=1, act=False, bn=False)
        self.pool = nn.Identity() if stride==1 else nn.AvgPool2d(2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(nf, affine=True) if bn else nn.Identity()
        
    def forward(self, x):
        return self.bn(F.relu(self.convs(x) + self.idconv(self.pool(x))))
    
    def __iter__(self):
        yield self.convs
        yield self.idconv
        yield self.pool
        yield self.bn
        
    
def acc_fn(out, yb):
    valid_mask_count = (yb != 255).sum()
    return (torch.argmax(out, dim=1) == yb).float().sum() / valid_mask_count

class CustomUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = conv(3, 64, ks=7, stride=1)
        self.down1 = ResBlock(64, 128)
        self.down2 = ResBlock(128, 256)
        self.down3 = ResBlock(256, 512)

        self.bottom = conv(512, 1024, ks=3, stride=1)
        self.bottom2 = conv(1024, 512, ks=3, stride=1)

        self.r1 = conv(192, 5, ks=3, stride=1)
        self.r2 = conv(384, 128, ks=3, stride=1)
        self.r3 = conv(768, 256, ks=3, stride=1)

    def forward(self, x): # (bs, 3, 384, 512)
        x = self.l1(x) # (bs, 64, 384, 512)
        x2 = self.down1(x) # (bs, 128, 192, 256)
        x3 = self.down2(x2) # (bs, 256, 96, 128)
        x4 = self.down3(x3) # (bs, 512, 48, 64)

        r4 = self.bottom2(self.bottom(x4)) # (bs, 512, 48, 64)
        r3 = F.interpolate(r4, scale_factor=2, mode="nearest") # (bs, 512, 96, 128)
        r3 = torch.cat([r3, x3], dim=1) # (bs, 768, 96, 128)
        r3 = self.r3(r3)
        r2 = F.interpolate(r3, scale_factor=2, mode="nearest")
        r2 = torch.cat([r2, x2], dim=1)
        r2 = self.r2(r2)
        r1 = F.interpolate(r2, scale_factor=2, mode="nearest")
        r1 = torch.cat([r1, x], dim=1) # (bs, 192, 384, 512)
        r1 = self.r1(r1) # (bs, 5, 384, 512)
        return r1

class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
        )
        # self.model = CustomUnet()
    
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
        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True)
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



