import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

import lightning as L
from transformers import SegformerForSemanticSegmentation


def acc_fn(out, mask):
    valid_mask_count = (mask != -100).sum()
    return (torch.argmax(out, dim=1) == mask).float().sum() / valid_mask_count
id_to_label = {
    0: 'road',
    1: 'lane_marking',
    2: 'undrivable',
    3: 'movable',
    4: 'my_car',
}

label_to_id = {v: k for k, v in id_to_label.items()}

class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/{config['backbone']}", 
            return_dict=False, 
            num_labels=5,
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
)
    
    def forward(self, x):
        low_res_mask = self.model(x)[0]
        h, w = low_res_mask.shape[-2:]
        return F.interpolate(low_res_mask, size=(4*h, 4*w), mode='bilinear', align_corners=False)
    
    def forward_val(self, x):
        # We do square crops during training. For inference we take the left and right crop
        # Would be simpler but there is some overlap
        pred_width = x.shape[-2]
        pred_height = x.shape[-2]
        imgs_left = x[:, :, :pred_height, :pred_width]
        imgs_right = x[:, :, :pred_height, -pred_width:]
        pred_left = self(imgs_left)
        pred_right = self(imgs_right)
        pred_left = F.interpolate(pred_left, size=(pred_height, pred_width), mode='bilinear', align_corners=False)
        pred_right = F.interpolate(pred_right, size=(pred_height, pred_width), mode='bilinear', align_corners=False)
        final_output = torch.zeros((x.shape[0], 5, x.shape[2], x.shape[3]), device=x.device)
        final_output[..., :pred_width] += pred_left
        final_output[..., -pred_width:] += pred_right
        final_output[..., -pred_width:pred_width] /= 2 # Average the overlap
        return final_output

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return {'loss': loss, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self.forward_val(x)

        if self.config["TTA"]:
            x_flipped = torch.flip(x, [3])
            out_flipped = self.forward_val(x_flipped)
            out_flipped = torch.flip(out_flipped, [3])
            out = (out + out_flipped) / 2

        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'accuracy': accuracy}

    def configure_optimizers(self):
        print("Configuring optimizers")
        norm_params = []
        encoder_params = []
        decoder_params = []
        for name, param in self.model.segformer.named_parameters():
            if "norm" in name:
                norm_params.append(param)
            else:
                encoder_params.append(param)
        for name, param in self.model.decode_head.named_parameters():
            if "norm" in name:
                norm_params.append(param)
            else:
                decoder_params.append(param)
        optimizer_groups = [
            {'params': encoder_params},
            {'params': norm_params, 'weight_decay': 0.0},
            {'params': decoder_params, 'lr': self.config['lr'] * 10},
        ]

        optimizer = torch.optim.AdamW(optimizer_groups, lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        devices = self.config["gpus"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.config['lr'],
                                                        # Steps per epoch is a bit harder to get since we need to wait for the dataloader to be created
                                                        steps_per_epoch = (len(self.trainer.fit_loop._data_source.dataloader())+devices-1)//devices,
                                                        epochs=self.config['epochs'],
                                                        pct_start=self.config['warmup_epochs']/self.config["epochs"],
                                                        anneal_strategy='linear')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }



