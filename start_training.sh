#!/bin/bash
source .venv/bin/activate

# Check for multi-seed flag
if [ "$1" == "--multi-seed" ]; then
    echo "Running multi-seed training (seeds: 42, 123, 456)..."
    python3 train_models_multi_seed.py
else
    echo "Running single-seed training (seed: 42)..."
    python3 train_models_bilstm.py
    python3 train_models_bert.py
    python3 train_models_roberta.py
fi