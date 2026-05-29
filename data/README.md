# Data Directory

Place downloaded and extracted datasets in this directory.

Recommended layout:

```text
data/
├── train/
└── test/
```

Suggested usage:

- Put the Zenodo training dataset under `data/train/`
- Put the Zenodo testing dataset under `data/test/`
- Use `data/test/` for the prepared CSV and image folders consumed by `evaluate/eval_model.py`

Zenodo sources:

- Training dataset: https://zenodo.org/records/15823641
- Testing dataset: https://zenodo.org/records/16034987
