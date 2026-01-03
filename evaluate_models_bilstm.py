#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import Resources.utils_stripped as U


def fbeta_from_pr(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = {}
    for avg in ["macro", "weighted", "micro"]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)
        metrics[f"f2_{avg}"] = float(fbeta_from_pr(float(p), float(r), beta=2.0))

    metrics["classification_report"] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def main():
    print("Starting BiLSTM model evaluation...")

    # Keep label mapping consistent with training
    train_df = pd.read_csv("Input/opp115_train.csv")
    test_df = pd.read_csv("Input/opp115_test.csv")
    train_df, test_df = U.load_opp115_from_df(train_df, test_df)

    sentences = test_df["Sentence"].astype(str).tolist()
    y_true = test_df["label"].astype(int).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load BiLSTM artifacts (project-specific) ----
    # You MUST load the exact same vocabulary/vectorizer used during training.
    # Common patterns in projects:
    # - a torch model file: model.pt / bilstm.pt
    # - a vocab/vectorizer pickle/json: vocab.pkl / vectorizer.pkl
    #
    # Replace these filenames with what your training actually saved.
    model_dir = "artifacts/artifacts/bilstm"  # fix your double "artifacts/artifacts" unless it's real
    model_path = os.path.join(model_dir, "model.pt")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    # If your utils already has a loader, use it.
    # Example names; adjust to your actual utils functions.
    # bilstm, vectorizer = U.load_bilstm(model_path, vectorizer_path, device=device)

    # If you trained and still have the model class in code, you can load state_dict:
    # bilstm = U.build_bilstm_from_artifacts(model_dir).to(device)
    # bilstm.load_state_dict(torch.load(model_path, map_location=device))

    # Best case: your utils already returns a ready-to-use model (as in earlier snippets)
    bilstm, vectorizer = U.load_bilstm_artifacts(model_dir, device=device)  # <-- rename to real function

    # ---- Predict using your BiLSTM inference function ----
    with U.measure_run("EVAL - BiLSTM - OPP115"):
        y_pred, _ = U.predict_bilstm(bilstm, sentences, return_proba=True)

    # Ensure integer label ids for sklearn
    if isinstance(y_pred[0], str):
        # If your predict returns label strings, map them back to ids exactly like training did.
        # Prefer: use mapping saved during training (label2id.json).
        label2id = U.load_label2id(model_dir)  # <-- rename to real function
        y_pred = np.array([label2id[s] for s in y_pred], dtype=int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    metrics = compute_metrics(y_true, y_pred)

    os.makedirs(model_dir, exist_ok=True)
    metrics_path = os.path.join(model_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(model_dir, "eval_predictions.csv")
    out_df = test_df.copy()
    out_df["y_pred"] = y_pred
    out_df.to_csv(pred_path, index=False)

    print("Saved:", metrics_path)
    print("Saved:", pred_path)
    print("Macro F1:", metrics["f1_macro"])
    print("Macro F2:", metrics["f2_macro"])
    print("Weighted F1:", metrics["f1_weighted"])
    print("Weighted F2:", metrics["f2_weighted"])


if __name__ == "__main__":
    main()
