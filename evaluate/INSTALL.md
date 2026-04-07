# Evaluation Dependencies Installation

## Required Package: onnxruntime

The evaluation script requires `onnxruntime` for model inference.

### Installation Options

**For GPU support (recommended):**
```bash
pip install onnxruntime-gpu
```

**For CPU only:**
```bash
pip install onnxruntime
```

### Verify Installation

```bash
python -c "import onnxruntime; print('✓ onnxruntime installed')"
```

### Check CUDA Support

```bash
python -c "import onnxruntime; print(onnxruntime.get_device())"
# Should print: 'GPU' if CUDA is available
```

## Optional: RDKit for SMILES Metrics

For advanced evaluation metrics (fingerprint similarity, etc.):

```bash
conda install -c conda-forge rdkit
```

---

After installing dependencies, run evaluation:

```bash
python evaluate/eval_model.py \
    --weights weights/I2M_R4.onnx \
    --dataset acs
```
