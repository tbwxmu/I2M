# I2M

I2M converts molecular structure images into chemical structure strings.
## No-code web app:
- BondlifeAI: https://www.bondlifeai.cn/

## Quick Start

Set up the environment and run inference on a single molecule image:

```bash
# 1. Clone the repository
git clone https://github.com/tbwxmu/I2M.git
cd I2M

# 2. Create the environment
conda env create -f environment.yml
conda activate i2m

# 3. Install the package
pip install -e .

# 4. Predict a SMILES string from one image
python infer_single.py test/test_001.png
```

The command prints the predicted SMILES string to stdout.

## Installation

- Linux or WSL environment
- NVIDIA GPU with CUDA support (recommended)
- Conda package manager
- Python 3.10+

The repository includes standard Python packaging metadata, so it can be installed with `pip`.

```bash
# Local source tree
pip install .

# Editable install for development
pip install -e .

# Include optional extras when needed
pip install ".[eval,ocr]"
```

you can install directly from GitHub as the repository is published:

```bash
pip install "git+https://github.com/tbwxmu/I2M.git"
```

Current packaging constraints:

- Python `3.10` only
- Linux only
- `paddlepaddle` is not installed automatically because the correct package depends on your CUDA / platform combination

## Inference

```bash
python infer_single.py test/test_001.png
```

Sample test images are included under [`test/`](./test), for example:

```bash
python infer_single.py test/test_001.png
python infer_single.py test/test_010.png
```

Optional arguments:

- `--weights`: use a different ONNX checkpoint
- `--threshold`: adjust the confidence cutoff before graph reconstruction
- `--device`: choose `cpu`, `cuda`, or `auto`
- `--print-detections`: print filtered detections to stderr before SMILES reconstruction

Dataset-oriented evaluation is still available:

```bash
python evaluate/eval_model.py \
    --resume weights/I2M_R4.onnx \
    --dataname acs \
    --data-root data/test
```

This script is designed for dataset evaluation rather than ad hoc single-image use. It expects a CSV file plus an image directory under `--data-root`.

## Data & PyTorch training checkpoints 

Download the official Zenodo datasets and place the extracted files under the repository's [`data/`](/recovery/bo/pys/I2M/data) directory.

- Training dataset: [Zenodo record 15823641](https://zenodo.org/records/15823641)， 
- Testing dataset: [Zenodo record 16034987](https://zenodo.org/records/16034987)
- PyTorch training checkpoints: [Zenodo record 20442887](https://zenodo.org/records/20442887)

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
- Use `data/test/` for the CSV and image folders consumed by `evaluate/eval_model.py`

If you evaluate custom benchmark splits with [`evaluate/eval_model.py`](/recovery/bo/pys/I2M/evaluate/eval_model.py), point `--data-root` to the directory containing the prepared CSV and image folders. The directory conventions are also documented in [`data/README.md`](/recovery/bo/pys/I2M/data/README.md).

## Training and Benchmarks

- Training guide: [`TRAINING.md`](/recovery/bo/pys/I2M/TRAINING.md)
- Evaluation setup: [`evaluate/INSTALL.md`](/recovery/bo/pys/I2M/evaluate/INSTALL.md)
- Data download helper: [`evaluate/download_data_guide.sh`](/recovery/bo/pys/I2M/evaluate/download_data_guide.sh)
- Supported benchmark datasets include ACS, JPO, UOB, USPTO, CLEF, Staker, and ChemVLOCR


## License

This project is released under the common noncommercial software license **PolyForm Noncommercial 1.0.0**. See [`LICENSE`](/recovery/bo/pys/I2M/LICENSE) for the full terms.

Commercial use is not permitted under the current license.

If you are interested in commercial licensing or collaboration, please contact `wuzxmu@gmail.com`.
