# comma10k-better

We iterate on the [comma10k-baseline](https://github.com/YassineYousfi/comma10k-baseline) codebase training a segnet on a [comma10k](https://github.com/commaai/comma10k) dataset. 

Using a segformer b3 we achieve a validation loss of 0.0345 with test time augmentations

## Training 
Similarly to baseline repo we use two step training

```
python main.py --batch_size 16 --epochs 40 --lr 2e-4 --width 582 --height 437 --warmup_epochs 2 --backbone mit-b3
python main.py --batch_size 4 --gpus 4 --epochs 22 --lr 6e-5 --warmup_epochs 2 --backbone mit-b3 --checkpoint_path <stage1.ckpt>

# Evaluate the model
python eval.py --batch_size 4 --checkpoint_path <stage2.ckpt> --TTA
```

The model takes 2 hours for [stage 1](https://wandb.ai/grekiki-squad/C10k-better/runs/10h67trv) on an rtx 3090 and 1.5h for [stage 2](https://wandb.ai/grekiki-squad/C10k-better/runs/t63e7r4u) on 4 rtx 3090s.


## Failures
- SAM(Segment Anything Model) finetuning was worse than the baseline
- [autoalbument](https://albumentations.ai/docs/autoalbument/) gave out weak augmentations even with about a day of training time


