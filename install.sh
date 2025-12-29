#!/bin/bash

# ============================================
# Face Recognition System - Unix Installer
# ============================================

echo "============================================"
echo "  FACE RECOGNITION SYSTEM INSTALLER"
echo "  Platform: $(uname -s)"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found!"
    echo "macOS: Install from python.org or use 'brew install python3'"
    echo "Linux: Use 'sudo apt install python3 python3-pip'"
    exit 1
fi

echo "[INFO] Python found!"
python3 --version

# Detect macOS and check for Apple Silicon
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[INFO] macOS detected"
    if [[ "$(uname -m)" == "arm64" ]]; then
        echo "[INFO] 🍎 Apple Silicon (M1/M2/M3) detected"
        echo "[INFO] Ensure you're using native ARM Python:"
        file $(which python3)
    fi
fi

# Create virtual environment
echo ""
echo "[INFO] Creating virtual environment..."
python3 -m venv venv

# Activate venv
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "[INFO] Creating project directories..."
mkdir -p dataset models logs utils

# Make scripts executable
chmod +x install.sh

echo ""
echo "============================================"
echo "  INSTALLATION COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Run: source venv/bin/activate"
echo "  2. Run: python main_system.py"
echo ""
echo "============================================"
