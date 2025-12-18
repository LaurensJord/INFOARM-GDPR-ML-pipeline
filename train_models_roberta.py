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

df_train, df_val = U.split_train_val(train_df, val_size=0.2)

with U.measure_run("Train BiLSTM"):
    res = U.train_bilstm_mcc(
        df_train=df_train,
        df_val=df_val,
        epochs=5,
        batch_size=32,
        lr=1e-4,
        out_dir="artifacts/bilstm"
    )

bilstm = res["model"]

# ## BiLSTM


print("test_df rows:", len(test_df))
print("train_df rows:", len(train_df))
print("Sentence non-null:", test_df["Sentence"].notna().sum())
print("Unique labels:", test_df["label"].nunique())

bilstm.eval()

sentences = test_df["Sentence"].tolist()
y_true = test_df["label"].astype(int).to_numpy()

with U.measure_run("BiLSTM inference"):
    y_pred_names, _ = U.predict_bilstm(bilstm, sentences, return_proba=True)

# map string labels → ids
CLASS_TO_ID = {name: i for i, name in enumerate(U.LABEL_NAMES)}
y_pred = np.array([CLASS_TO_ID[name] for name in y_pred_names], dtype=int)

results = U.evaluate_predictions(y_true, y_pred)

print("Confusion matrix:")
print(results["confusion_matrix"])

p = results["report"]["macro avg"]["precision"]
r = results["report"]["macro avg"]["recall"]

print("Macro F1:", results["report"]["macro avg"]["f1-score"])
print("Macro F2:", 0.0 if (4*p + r) == 0 else (5*p*r)/(4*p + r))