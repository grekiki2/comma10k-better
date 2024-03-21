# comma10k-better

We iterate on the [comma10k-baseline](https://github.com/YassineYousfi/comma10k-baseline) codebase training a segnet on a [comma10k](https://github.com/commaai/comma10k) dataset. 

By removing weight decay, using a higher learning rate with warmup on 3x fewer epochs, changing out Unet for DeeplabV3+ with a smaller backbone and using test time augmentations the code achieves a validation loss of 0.0396. 

## Training 
Similarly to baseline repo we use two step training

```
python main.py --batch_size 16 --epochs 20 --lr 4e-4 --width 582 --height 437 --warmup_epochs 2
python main.py --batch_size 8 --epochs 22 --lr 1e-4 --warmup_epochs 2 --checkpoint_path <stage1.ckpt>

# Evaluate the model
python eval.py --batch_size 8 --checkpoint_path <stage2.ckpt> --TTA
```

The model takes 5 hours to train on an RTX 3090 on Python 3.11.4

## Improvement ideas

- With more gpus/VRAM an efficientnet-b4 backbone could be tested with higher batch sizes. 
- torch.compile doesn't seem to work nicely with `segmentation-models-pytorch`
- lsuv could be used for better fine-tuning initialization
- SWA was attempted but didn't give any improvement, might be a skill issue. 