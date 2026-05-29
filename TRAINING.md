# Training I2M

This document collects the training-focused workflow that was previously in the main README.

## Environment Setup

```bash
bash setup_i2m.sh
conda activate i2m
```

Or install manually:

```bash
conda env create -f environment.yml
conda activate i2m
```

## Start Training

```bash
python tools/train.py --config configs/moldetr/moldetr_r50vd_6x_coco.yml
```

## Basic Training

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --output_dir output/my_first_model
```

## Fine-Tuning

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --tuning pretrained/model.pth \
    --output_dir output/fine_tuned
```

## Resume From Checkpoint

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --resume output/my_model/checkpoint0050.pth \
    --output_dir output/my_model
```

## Mixed Precision

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --amp \
    --output_dir output/amp_training
```

## Notes

- Update dataset paths in `configs/dataset/*.yml` before training.
- The training code assumes a Linux or WSL environment and Python 3.10.
- Pretrained weights and evaluation assets are stored separately from the training configs.
