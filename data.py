import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
import albumentations as A
import cv2
from pathlib import Path
import numpy as np

DATA_PATH = Path("~/Desktop/segnet/comma10k").expanduser()
CLASS_VALUES = [41,  76, 124, 161, 90]

class C10kDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        print(f"Setupping data for stage {stage}...")
        imgs = list((DATA_PATH / "imgs").glob("*.png"))
        masks = list((DATA_PATH / "masks").glob("*.png"))
        assert len(imgs) == len(masks), "Number of images and masks do not match"
        train_imgs = [x for x in imgs if not x.name.endswith("9.png")]
        train_masks = [x for x in masks if not x.name.endswith("9.png")]
        valid_imgs = [x for x in imgs if x.name.endswith("9.png")]
        valid_masks = [x for x in masks if x.name.endswith("9.png")]
        self.train_ds = DatasetWrapper(train_imgs, train_masks)
        self.val_ds = DatasetWrapper(valid_imgs, valid_masks)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=16, pin_memory=True)


class DatasetWrapper(Dataset):
    def __init__(self, img_paths, mask_paths, dim=(384, 512)):
        assert len(img_paths) == len(mask_paths), "Number of images and masks do not match"
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.dim = dim
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (1164, 874, 3) np.uint8
        img_rgb = (img_rgb / 255.0).astype(np.float32) # (1164, 874, 3) np.float32

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE) # (1164, 874)
        conditions = [mask == v for v in CLASS_VALUES]
        mask_ids = np.select(conditions, list(range(5))) # (1164, 874) but with 0-4 values. CLASS_VALUES were determined by .ipynb

        img_rgb = cv2.resize(img_rgb, self.dim)
        mask_ids = cv2.resize(mask_ids, self.dim, interpolation=cv2.INTER_NEAREST)

        return np.transpose(img_rgb, (2, 0, 1)), mask_ids.astype(np.int64)