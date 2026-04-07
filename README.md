# I2M 

A streamlined deep learning model for converting molecular structure images to chemical structures using DETR architecture.

**Project Size**: ~1 MB | **Files**: 87 | **Status**: ✅ Ready for Training

## Quick Start

Get started：

```bash
# 1. Setup environment
bash setup_i2m.sh && conda activate i2m

# 2. Start training
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
---

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

**Note**: Evaluation data is NOT included in this repository. See `evaluate/README.md` or related paper for data download instructions.

---

## License

MIT License - See LICENSE file for details.


