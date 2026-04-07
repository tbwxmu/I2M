# I2M (Image to Molecule) - Minimal Edition

A streamlined deep learning model for converting molecular structure images to chemical structures using RT-DETR architecture.

**Project Size**: ~1 MB | **Files**: 87 | **Status**: ✅ Ready for Training

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Tips for Better Results](#tips-for-better-results)

---

## Quick Start

Get started in 3 commands:

```bash
# 1. Setup environment
bash setup_i2m.sh && conda activate i2m

# 2. Prepare data structure
python create_sample_dataset.py

# 3. Start training
python tools/train.py --config configs/moldetr/moldetr_r50vd_6x_coco.yml
```

---

## Installation

### Prerequisites

- Linux or WSL environment
- NVIDIA GPU with CUDA support (recommended)
- Conda package manager
- Python 3.10+

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/I2M.git
cd I2M
```

2. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate i2m
```

3. **Verify installation**:
```bash
python verify_setup.py
```

### Manual Installation (if needed)

If `setup_i2m.sh` doesn't work, install manually:

```bash
# Configure pip mirror (China users)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install core dependencies
conda install -y pytorch torchvision torchaudio pyyaml -c conda-forge
conda install -y rdkit pandas numpy opencv pillow scipy scikit-learn matplotlib tqdm

# Install additional packages
pip install paddleocr paddlepaddle-gpu SmilesPE cairosvg pycocotools
```

---

## Data Preparation

I2M expects data in COCO format. You have two options:

### Option A: Use Your Own Dataset

Create the following directory structure:

```
data/custom_training/
├── images/
│   ├── train/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── val/
│       ├── img1.png
│       ├── img2.png
│       └── ...
└── annotations/
    ├── train.json
    └── val.json
```

Update `configs/dataset/coco_detection.yml`:

```yaml
train_dataloader:
  dataset:
    img_folder: data/custom_training/images/train
    ann_file: data/custom_training/annotations/train.json

val_dataloader:
  dataset:
    img_folder: data/custom_training/images/val
    ann_file: data/custom_training/annotations/val.json
```

### Option B: Generate from SMILES/Molecules

Use the provided script to create data structure:

```bash
python create_sample_dataset.py
```

This creates the directory template and an example annotation file.

### COCO Annotation Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "molecule_001.png",
      "width": 512,
      "height": 512
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 50, 50],
      "area": 2500,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "carbon", "supercategory": "atom"},
    {"id": 2, "name": "oxygen", "supercategory": "atom"}
  ]
}
```

---

## Training

### Basic Training (from scratch)

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --output_dir output/my_first_model
```

### Fine-tuning from Pre-trained Model

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --tuning pretrained/model.pth \
    --output_dir output/fine_tuned
```

### Resume Training from Checkpoint

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --resume output/my_model/checkpoint0050.pth \
    --output_dir output/my_model
```

### Mixed Precision Training (Faster)

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --amp \
    --output_dir output/amp_training
```

### Monitor Training

```bash
# Watch logs in real-time
tail -f output/my_model/train.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check progress
grep "Epoch" output/my_model/train.log | tail -20
```

### Expected Training Time

- Single GPU (RTX 3090): ~2-3 days
- Single GPU (RTX 4090): ~1-2 days
- Multi-GPU (4x): ~6-12 hours

---

## Inference

### Run Inference on Test Images

```bash
python tools/infer.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --resume output/my_model/best_checkpoint.pth \
    --test-only \
    --infer \
    --csv_path data/test.csv \
    --image_dir data/test/images \
    --gpuid 0 \
    --outcsv_filename output/predictions.csv
```

### Evaluate on Benchmark Datasets

I2M includes evaluation scripts for standard molecular structure recognition benchmarks.

**Available datasets**: ACS, JPO, UOB, USPTO, CLEF, Staker, ChemVLOCR

```bash
# Run evaluation
python evaluate/eval_model.py \
    --weights weights/I2M_R4.onnx \
    --dataset acs \
    --output_dir evaluate/results
```

**Note**: Evaluation data is NOT included in this repository. See `evaluate/README.md` for data download instructions.

For detailed evaluation guide, see: [evaluate/README.md](evaluate/README.md)

### Export to ONNX

```bash
python tools/train.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --resume output/my_model/best_checkpoint.pth \
    --test-only
```

### Export to ONNX

```bash
python tools/export_onnx.py \
    --config configs/moldetr/moldetr_r50vd_6x_coco.yml \
    --resume output/my_model/best_checkpoint.pth \
    --output_dir output/onnx_model
```

---

## Configuration

### Key Configuration Files

| File | Purpose | When to Edit |
|------|---------|--------------|
| `configs/dataset/coco_detection.yml` | Data paths & dataloader | Before training |
| `configs/moldetr/moldetr_r50vd_6x_coco.yml` | Model architecture & optimizer | Optional |
| `configs/runtime.yml` | AMP, EMA settings | For advanced tuning |

### Adjust Batch Size (if OOM)

Edit `configs/dataset/coco_detection.yml`:

```yaml
train_dataloader:
  batch_size: 8  # Reduce from 16 if out of memory
  num_workers: 8  # Adjust based on CPU cores
```

### Adjust Learning Rate

Edit `configs/moldetr/moldetr_r50vd_6x_coco.yml`:

```yaml
optimizer:
  lr: 0.0001  # Smaller for fine-tuning
```

---

## Project Structure

```
I2M/
├── configs/                 # Configuration files
│   ├── dataset/            # Dataset configurations
│   │   ├── coco_detection.yml          # ← Edit this for your data
│   │   └── coco_detection_custom.yml   # Template
│   ├── moldetr/             # Model configurations
│   │   └── moldetr_r50vd_6x_coco.yml    # Default model
│   └── runtime.yml         # Runtime settings
├── src/                    # Source code (core framework)
│   ├── core/               # Config handling
│   ├── nn/                 # Neural network modules
│   ├── solver/             # Training logic
│   └── zoo/                # RT-DETR implementation
├── tools/                  # Essential scripts
│   ├── train.py            # ← Training script
│   ├── infer.py            # Inference
│   ├── export_onnx.py      # ONNX export
│   └── test.py             # Testing
├── evaluate/               # Evaluation scripts & benchmarks
│   ├── eval_model.py       # Model evaluation
│   └── README.md           # Evaluation guide
├── weights/                # Model weights (not in git)
│   └── README.md           # Weights placement guide
├── environment.yml         # Conda environment
├── setup_i2m.sh           # One-click setup
├── verify_setup.py        # Verify installation
├── create_sample_dataset.py  # Data structure template
└── README.md              # This file
```

**Note**: `data/` and `output/` directories are created when you prepare data and start training.

---

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size in `configs/dataset/coco_detection.yml`:
```yaml
train_dataloader:
  batch_size: 8  # or even 4
```

Or enable mixed precision:
```bash
python tools/train.py --config configs/moldetr/moldetr_r50vd_6x_coco.yml --amp
```

### Slow Training

**Solutions**:
- Enable mixed precision: add `--amp` flag
- Increase `num_workers` in config (if you have enough CPU cores)
- Use larger batch size (if GPU memory allows)

### Poor Accuracy

**Solutions**:
- Train for more epochs
- Use data augmentation
- Fine-tune from a pre-trained model
- Check data quality and annotations
- Adjust learning rate (try smaller LR like 0.0001 for fine-tuning)

### Import Errors

**Solution**: Make sure conda environment is activated:
```bash
conda activate i2m
python verify_setup.py
```

### Data Loading Errors

**Solutions**:
- Verify image paths in annotation files
- Check that all images exist
- Ensure COCO format is correct
- Run: `python verify_setup.py`

---

## Tips for Better Results

1. **Data Quality**: Ensure high-quality, diverse training data
2. **Data Augmentation**: Use various augmentations to improve generalization
3. **Learning Rate**: Start with smaller LR (0.0001) for fine-tuning
4. **Batch Size**: Use largest batch size your GPU can handle
5. **Training Duration**: Train until validation loss plateaus
6. **Checkpoint Selection**: Use `best_checkpoint.pth`, not the last one
7. **Mixed Precision**: Enable `--amp` for 2x faster training
8. **Monitor GPU**: Use `watch nvidia-smi` to check utilization

---

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review training logs: `output/*/train.log`
3. Verify installation: `python verify_setup.py`
4. Check GPU memory: `nvidia-smi`
5. Open a GitHub issue with:
   - Error message
   - Your configuration
   - System specs (GPU, RAM, OS)

---

## License

MIT License - See LICENSE file for details.

---

**Ready to start?** Run `bash setup_i2m.sh` to begin! 🚀
