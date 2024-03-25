import torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

import lightning as L
import pickle
import sys
sys.path.append("../repo")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

with open('tds_emb_dict.pkl', 'rb') as file:
    tds_emb_dict = pickle.load(file)

with open('vds_emb_dict.pkl', 'rb') as file:
    vds_emb_dict = pickle.load(file)

sam_model = sam_model_registry["vit_h"](checkpoint="../repo/sam_vit_h_4b8939.pth")
sam_model.train()
sparse_prompt_embeddings, dense_prompt_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
    )
dense_pos_emb = sam_model.prompt_encoder.get_dense_pe()
sparse_prompt_embeddings = sparse_prompt_embeddings.cuda()
dense_prompt_embeddings = dense_prompt_embeddings.cuda()
dense_pos_emb = dense_pos_emb.cuda()

def acc_fn(out, mask):
    valid_mask_count = (mask != -100).sum()
    return (torch.argmax(out, dim=1) == mask).float().sum() / valid_mask_count
d = (1164-874)//2
class Model(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.mask_decoder = sam_model.mask_decoder

        self.upscale1 = nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1)
        self.upscale2 = nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1)

        self.refinement = nn.Sequential(
            nn.Conv2d(6, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 5, 1),
        )

    
    def forward(self, fnames, x, train=True):
        dct = tds_emb_dict if train else vds_emb_dict
        embs = torch.cat([torch.tensor(dct[fname]) for fname in fnames], dim=0).cuda()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=embs,
            image_pe=dense_pos_emb,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=True,
        )
        masks = F.relu(self.upscale1(low_res_masks))
        masks = F.relu(self.upscale2(masks))
        upscaled_masks = F.interpolate(masks, size=(1164, 1164), mode='bilinear', align_corners=False)
        x = F.pad(x, (0, 0, d, d))
        refinement_input = torch.cat([x, upscaled_masks], dim=1)
        return self.refinement(refinement_input)[...,d:-d,:]
    
    def training_step(self, batch, batch_idx):
        fnames, x, y = batch
        out = self(fnames, x, True)
        loss = F.cross_entropy(out, y)
        accuracy = acc_fn(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return {'loss': loss, 'accuracy': accuracy}
    
    def validation_step(self, batch, batch_idx):
        fnames, x, y = batch
        out = self(fnames, x, False)
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



