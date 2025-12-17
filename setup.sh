#!/bin/bash

# Exit on error
set -e

echo "Setting up INFOARM-GDPR-ML-pipeline environment..."

# =============================================================================
# Create virtual environment
# =============================================================================
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip setuptools wheel

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# =============================================================================
# Download and setup datasets
# =============================================================================
echo ""
echo "Setting up datasets..."

# Create Input directory if it doesn't exist
mkdir -p Input

# Download OPP-115 dataset
OPP115_URL="https://www.usableprivacy.org/static/data/OPP-115_v1_0.zip"
OPP115_ZIP="Input/OPP-115.zip"
OPP115_DIR="Input/OPP-115"

if [ ! -d "$OPP115_DIR" ]; then
    echo "Downloading OPP-115 dataset (~95MB)..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$OPP115_ZIP" "$OPP115_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$OPP115_ZIP" "$OPP115_URL"
    else
        echo "Error: wget or curl is required to download datasets"
        exit 1
    fi
    
    echo "Extracting OPP-115 dataset..."
    unzip -q "$OPP115_ZIP" -d "Input/"
    
    # Clean up zip file
    rm -f "$OPP115_ZIP"
    
    # Remove MacOS artifacts if present
    rm -rf "Input/__MACOSX"
    
    echo "OPP-115 dataset ready at $OPP115_DIR"
else
    echo "OPP-115 dataset already exists at $OPP115_DIR"
fi

# Download JURIX 2020 GDPR Mapping dataset
JURIX_URL="https://www.usableprivacy.org/static/data/JURIX_2020_OPP-115_GDPR_v1.0.zip"
JURIX_ZIP="Input/JURIX_2020_mapping.zip"
JURIX_DIR="Input/jurix_2020_opp-115_gdpr_dataset"

if [ ! -d "$JURIX_DIR" ]; then
    echo "Downloading JURIX 2020 GDPR mapping dataset..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$JURIX_ZIP" "$JURIX_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$JURIX_ZIP" "$JURIX_URL"
    else
        echo "Error: wget or curl is required to download datasets"
        exit 1
    fi
    
    echo "Extracting JURIX 2020 dataset..."
    unzip -q "$JURIX_ZIP" -d "Input/"
    
    # Clean up zip file
    rm -f "$JURIX_ZIP"
    
    echo "JURIX 2020 dataset ready at $JURIX_DIR"
else
    echo "JURIX 2020 dataset already exists at $JURIX_DIR"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Datasets available in Input/:"
echo "  - OPP-115/ (115 privacy policies with annotations)"
echo "  - jurix_2020_opp-115_gdpr_dataset/ (GDPR mapping)"
echo ""
echo "To preprocess the datasets for GDPR classification, run:"
echo "  python Resources/data_preprocessing.py Input/OPP-115"
echo ""
echo "  - opp115_test.csv (preprocessed test data)"
echo ""
