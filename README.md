# I2M

I2M converts molecular structure images into chemical structure strings.

## Quick Start

Set up the project and run inference on a single molecule image:

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate i2m

# 2. Install the package
pip install -e .

# 3. Predict a SMILES string from one image
python infer_single.py test/test_001.png
```

The command prints the predicted SMILES string to stdout.

If you prefer a no-code workflow, a web app frontend is also available:

- BondlifeAI web app: https://www.bondlifeai.cn/

## Main Entry Point

[`infer_single.py`](/recovery/bo/pys/I2M/infer_single.py) is the simplest image-to-structure entry point in this repository. It:

- runs the ONNX model in [`weights/I2M_R4.onnx`](/recovery/bo/pys/I2M/weights/I2M_R4.onnx)
- prints the reconstructed SMILES string

## Installation

### Prerequisites

- Linux or WSL environment
- NVIDIA GPU with CUDA support (recommended)
- Conda package manager
- Python 3.10+

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone https://github.com/tbwxmu/I2M.git
cd I2M
```

2. **Create conda environment**:
```bash
conda env create -f environment.yml
conda activate i2m
```

### Install with pip

The repository includes standard Python packaging metadata, so it can be installed with `pip`.

```bash
# Local source tree
pip install .

# Editable install for development
pip install -e .

# Include optional extras when needed
pip install ".[eval,ocr]"
```

If you want to install directly from GitHub after the repository is published:

```bash
pip install "git+https://github.com/tbwxmu/I2M.git"
```

Current packaging constraints:

- Python `3.10` only
- Linux only
- The package includes a prebuilt binary module: `src/solver/det_engine.cpython-310-x86_64-linux-gnu.so`
- `paddlepaddle` is not installed automatically because the correct package depends on your CUDA / platform combination

## Inference

### Single image

```bash
python infer_single.py test/test_001.png
```

Sample test images are included under [`test/`](/recovery/bo/pys/I2M/test), for example:

```bash
python infer_single.py test/test_001.png
python infer_single.py test/test_010.png
```

Optional arguments:

- `--weights`: use a different ONNX checkpoint
- `--threshold`: adjust the confidence cutoff before graph reconstruction
- `--device`: choose `cpu`, `cuda`, or `auto`
- `--print-detections`: print filtered detections to stderr before SMILES reconstruction

### Benchmark or batch-style evaluation

The original dataset-oriented evaluation entry point is still available:

```bash
python evaluate/eval_model.py \
    --resume weights/I2M_R4.onnx \
    --dataname acs \
    --data-root data/test
```

This script is designed for dataset evaluation rather than ad hoc single-image use. It expects a CSV file plus an image directory under `--data-root`.

### No-code web app

If you do not want to run the local environment, you can use the web frontend:

- https://www.bondlifeai.cn/

This provides a zero-code way to use I2M in the browser.

## Data

Download the official Zenodo datasets and place the extracted files under the repository's [`data/`](/recovery/bo/pys/I2M/data) directory.

- Training dataset: [Zenodo record 15823641](https://zenodo.org/records/15823641)
- Testing dataset: [Zenodo record 16034987](https://zenodo.org/records/16034987)

Recommended layout:

```text
I2M/
├── data/
│   ├── train/
│   ├── test/
├── test/
│   ├── test_001.png
│   └── ...
└── ...
```

Suggested usage:

- Put the extracted training dataset under `data/train/`
- Put the extracted testing dataset under `data/test/`
- Use `data/test/` for the CSV and image folders consumed by `evaluate/eval_model.py`

If you evaluate custom benchmark splits with [`evaluate/eval_model.py`](/recovery/bo/pys/I2M/evaluate/eval_model.py), point `--data-root` to the directory containing the prepared CSV and image folders. The directory conventions are also documented in [`data/README.md`](/recovery/bo/pys/I2M/data/README.md).

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

## Training

Training documentation has been moved to [`TRAINING.md`](/recovery/bo/pys/I2M/TRAINING.md).

Quick link:

```bash
python tools/train.py --config configs/moldetr/moldetr_r50vd_6x_coco.yml
```

## Publish to PyPI

Before uploading, make sure the package name `I2M` is still available on PyPI.

```bash
# 1. Build source + wheel
python -m build

# 2. Check the generated artifacts
python -m twine check dist/*

# 3. Upload to TestPyPI first
python -m twine upload --repository testpypi dist/*

# 4. Upload to PyPI
python -m twine upload dist/*
```

If the name `I2M` is already taken on PyPI, change `project.name` in [`pyproject.toml`](/recovery/bo/pys/I2M/pyproject.toml) to a unique distribution name such as `i2m-moldetr`.

## Benchmarks

I2M includes evaluation scripts for benchmark datasets such as ACS, JPO, UOB, USPTO, CLEF, Staker, and ChemVLOCR.

Evaluation data is not included in this repository. Download it from Zenodo, extract it into [`data/`](/recovery/bo/pys/I2M/data), and refer to [`evaluate/INSTALL.md`](/recovery/bo/pys/I2M/evaluate/INSTALL.md) and [`evaluate/download_data_guide.sh`](/recovery/bo/pys/I2M/evaluate/download_data_guide.sh) for setup details.

---

## License

PolyForm Noncommercial 1.0.0. See [`LICENSE`](/recovery/bo/pys/I2M/LICENSE) for details. Commercial use is not permitted under the current license.
