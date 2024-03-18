"""
Pretvorimo dataset v np.array in jih shranimo. 
To bistveno pospeši zagon treninga kar pride prav za razhroscevanje
"""
import os
import random
from pathlib import Path

DATA_PATH = Path("~/Desktop/segnet/comma10k").expanduser()

if __name__ == '__main__':
    imgs = list((DATA_PATH / "imgs").glob("*.png"))
    masks = list((DATA_PATH / "masks").glob("*.png"))
    assert len(imgs) == len(masks), "Number of images and masks do not match"
    print(f"Found {len(imgs)} images and masks")
    train_imgs = [x for x in imgs if not x.name.endswith("9.png")]
    train_masks = [x for x in masks if not x.name.endswith("9.png")]
    valid_imgs = [x for x in imgs if x.name.endswith("9.png")]
    valid_masks = [x for x in masks if x.name.endswith("9.png")]
    print(f"Train: {len(train_imgs)} images and masks")
    print(f"Valid: {len(valid_imgs)} images and masks")
    train_imgs.sort()
    train_masks.sort()
    valid_imgs.sort()
    valid_masks.sort()




