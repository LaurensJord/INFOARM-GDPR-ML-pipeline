#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import torch
import time
import Resources.utils_stripped as U


train_df = pd.read_csv("Input/opp115_train.csv")
test_df = pd.read_csv("Input/opp115_test.csv")
train_df, test_df = U.load_opp115_from_df(train_df, test_df)
df_train, df_val = U.split_train_val(train_df, val_size=0.2)


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


# ## BERT Model

# Train BERT (multi-class)
res = U.train_transformer_mcc(
    model_type="bert",
    df_train=df_train,
    df_val=df_val,
    epochs=20,
    batch_size=64,
    lr=3e-5,
    out_dir="artifacts/bert_mcc"
)

bert_model = res["model"]
bert_tokenizer = res["tokenizer"]