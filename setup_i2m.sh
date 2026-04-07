#!/bin/bash
# Setup script for I2M project
# Run this after cloning from GitHub to get started

set -e  # Exit on error

echo "=========================================="
echo "I2M Project Setup"
echo "=========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda found: $(which conda)"
echo ""

# Check if environment already exists
ENV_NAME="i2m"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠ Conda environment '${ENV_NAME}' already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME
    else
        echo "Using existing environment"
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo ""
echo "✓ Environment created successfully"
echo ""

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ""
echo "✓ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Verify installation
echo "Verifying installation..."
python verify_setup.py || {
    echo ""
    echo "⚠ Some verification checks failed"
    echo "You may need to install additional dependencies"
    echo ""
}

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment (if not already active):"
echo "   conda activate i2m"
echo ""
echo "2. Prepare your training data:"
echo "   python create_sample_dataset.py"
echo ""
echo "3. Update configuration files:"
echo "   - configs/dataset/coco_detection.yml (data paths)"
echo "   - configs/rtdetr/rtdetr_r50vd_6x_coco.yml (training parameters)"
echo ""
echo "4. Start training:"
echo "   python tools/train.py --config configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
echo ""
echo "For detailed instructions, see:"
echo "  - QUICK_START.md"
echo "  - TRAINING_GUIDE.md"
echo "  - README.md"
echo ""
echo "Happy training! 🚀"
echo ""
