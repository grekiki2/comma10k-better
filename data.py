from torch.utils.data import Dataset, DataLoader
import lightning as L
import albumentations as A
import cv2
from pathlib import Path
import numpy as np
import segmentation_models_pytorch as smp

DATA_PATH = Path("~/Desktop/segnet/comma10k").expanduser()
CLASS_VALUES = [41,  76, 124, 161, 90]

class C10kDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.height = config["height"]
        self.width = config["width"]
        self.backbone = config["backbone"]
        self.augmentation_level = config["augmentation_level"]
    
    def setup(self, stage=None):
        imgs = list((DATA_PATH / "imgs").glob("*.png"))
        masks = list((DATA_PATH / "masks").glob("*.png"))
        assert len(imgs) == len(masks), "Number of images and masks do not match"
        train_imgs = [x for x in imgs if not x.name.endswith("9.png")]
        train_masks = [x for x in masks if not x.name.endswith("9.png")]
        valid_imgs = [x for x in imgs if x.name.endswith("9.png")]
        valid_masks = [x for x in masks if x.name.endswith("9.png")]
        self.train_ds = DatasetWrapper(train_imgs[:5000], train_masks[:5000], None, None)
        self.val_ds = DatasetWrapper(valid_imgs, valid_masks, None, None)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8) # why does this crash with 16 workers?


class DatasetWrapper(Dataset):
    def __init__(self, img_paths, mask_paths, transform, backbone):
        assert len(img_paths) == len(mask_paths), "Number of images and masks do not match"
        self.img_paths = img_paths
        self.mask_paths = mask_paths
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (1164, 874, 3) np.uint8

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE) # (1164, 874)
        conditions = [mask == v for v in CLASS_VALUES]
        mask_ids = np.select(conditions, list(range(5))) # (1164, 874) with 0-4 values. CLASS_VALUES determined by .ipynb

        return self.img_paths[idx].name, np.transpose(img_rgb.astype(np.float32), (2, 0, 1)), mask_ids.astype(np.int64)