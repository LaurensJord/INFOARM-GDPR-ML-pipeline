#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import torch
import time
import Resources.utils_stripped as U


print("1. Starting training...")
print("2. Loading data...")
train_df = pd.read_csv("Input/opp115_train.csv")
test_df = pd.read_csv("Input/opp115_test.csv")
print("3. Data loaded")
train_df, test_df = U.load_opp115_from_df(train_df, test_df)
print("4. Labels processed")
df_train, df_val = U.split_train_val(train_df, val_size=0.2)
print("5. Split done")


# # Training of Models
# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Verify GPU compatibility
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("GPU Compute Capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "N/A")
print("Supported CUDA architectures:", torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else "N/A")

# Test a simple CUDA operation
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("✓ CUDA tensor test passed:", x)

# Train BiLSTM Model
with U.measure_run("TRAINING - BiLSTM - OPP115"):
    res = U.train_bilstm_mcc(
        df_train=df_train,
        df_val=df_val,
        epochs=3,
        batch_size=32,
        lr=1e-4,
        out_dir="artifacts/bilstm"
    )

bilstm = res["model"]