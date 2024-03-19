import torch
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
    def __init__(self, batch_size=32, height=437, width=582):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
    
    def setup(self, stage=None):
        print(f"Setupping data for stage {stage}...")
        imgs = list((DATA_PATH / "imgs").glob("*.png"))
        masks = list((DATA_PATH / "masks").glob("*.png"))
        assert len(imgs) == len(masks), "Number of images and masks do not match"
        train_imgs = [x for x in imgs if not x.name.endswith("9.png")]
        train_masks = [x for x in masks if not x.name.endswith("9.png")]
        valid_imgs = [x for x in imgs if x.name.endswith("9.png")]
        valid_masks = [x for x in masks if x.name.endswith("9.png")]
        self.train_ds = DatasetWrapper(train_imgs, train_masks, get_transforms(self.height, self.width, "hard"))
        self.val_ds = DatasetWrapper(valid_imgs, valid_masks, get_valid_transforms(self.height, self.width))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8) # why does this crash with 16 workers?

def pad_to_multiple(x, k=32):
    return int(k * (np.ceil(x / k)))

def get_scale_transform(height: int, width: int):
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.PadIfNeeded(
                pad_to_multiple(height),
                pad_to_multiple(width),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=-100, # cross entropy loss ignores -100
            ),
        ]
    )


def get_transforms(height: int, width: int, level: str):
    if level == "light":
        return A.Compose(
            [
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),
                        A.RandomBrightnessContrast(p=1.0, contrast_limit=0),
                        A.RandomGamma(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0, brightness_limit=0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.Resize(height=height, width=width, p=1.0),
            ],
            p=1.0,
        )
    elif level == "hard":
        return A.Compose(
            [
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                        A.ElasticTransform(
                            alpha_affine=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=0,
                            rotate_limit=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                        A.OpticalDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),
                        A.RandomBrightnessContrast(p=1.0, contrast_limit=0),
                        A.RandomGamma(p=1.0),
                        A.ISONoise(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0, brightness_limit=0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.Resize(height=height, width=width, p=1.0),
                A.CoarseDropout(p=0.3, mask_fill_value=-100),
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=-100,
                ),
            ],
            p=1.0,
        )
    elif level == "hard_weather":
        return A.Compose(
            [
                A.PadIfNeeded(
                    pad_to_multiple(height),
                    pad_to_multiple(width),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.GridDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                        A.ElasticTransform(
                            alpha_affine=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0,
                            scale_limit=0,
                            rotate_limit=10,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                        A.OpticalDistortion(
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=-100,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.CLAHE(p=1.0),
                        A.RandomBrightnessContrast(p=1.0, contrast_limit=0),
                        A.RandomGamma(p=1.0),
                        A.ISONoise(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(p=1.0, brightness_limit=0),
                        A.HueSaturationValue(p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_upper=0.8, p=1.0),
                        A.RandomRain(p=1.0),
                        A.RandomSnow(p=1.0),
                        A.RandomSunFlare(src_radius=100, p=1.0),
                    ],
                    p=0.4,
                ),
                A.Resize(height=height, width=width, p=1.0),
                A.CoarseDropout(p=0.3, mask_fill_value=-100),
            ],
            p=1.0,
        )
    else:
        raise ValueError

def get_valid_transforms(height: int, width: int):
    return get_scale_transform(height, width)

class DatasetWrapper(Dataset):
    def __init__(self, img_paths, mask_paths, transform):
        assert len(img_paths) == len(mask_paths), "Number of images and masks do not match"
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocess = smp.encoders.get_preprocessing_fn("efficientnet-b3")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (1164, 874, 3) np.uint8

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE) # (1164, 874)
        conditions = [mask == v for v in CLASS_VALUES]
        mask_ids = np.select(conditions, list(range(5))) # (1164, 874) with 0-4 values. CLASS_VALUES determined by .ipynb

        img_rgb, mask_ids = self.transform(image=img_rgb, mask=mask_ids).values()
        img_rgb = self.preprocess(img_rgb)

        return np.transpose(img_rgb.astype(np.float32), (2, 0, 1)), mask_ids.astype(np.int64)