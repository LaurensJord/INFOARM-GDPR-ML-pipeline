#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import Resources.utils_stripped as U
from Resources.utils_stripped import BiLSTMClassifier, EmbeddingConfig


def fbeta_from_pr(p: float, r: float, beta: float = 2.0) -> float:
    if p == 0.0 and r == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    metrics = {}
    for avg in ["macro", "weighted", "micro"]:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)
        metrics[f"f2_{avg}"] = float(fbeta_from_pr(float(p), float(r), beta=2.0))

    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def load_meta_and_state(model_dir: str):
    # zoekt *meta.json en *state.bin in model_dir
    meta_path = None
    state_path = None
    for fn in os.listdir(model_dir):
        if fn.endswith("_meta.json"):
            meta_path = os.path.join(model_dir, fn)
        if fn.endswith("_state.bin"):
            state_path = os.path.join(model_dir, fn)

    if meta_path is None or state_path is None:
        raise FileNotFoundError(f"Missing *_meta.json or *_state.bin in {model_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    state = torch.load(state_path, map_location="cpu")
    return meta, state


def build_bilstm_from_meta(meta: dict, state: dict):
    n_classes = int(meta.get("n_classes", meta.get("num_labels", 0)) or 0)
    input_size = meta.get("input_size", 384)
    embedding_model = meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_max_len = meta.get("embedding_max_len", 160)

    # Infer hidden_dim from state dict: lstm.weight_ih_l0 has shape [4*hidden_dim, input_size]
    hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4

    model = BiLSTMClassifier(
        input_size=input_size,
        hidden_dim=hidden_dim,
        output_dim=n_classes,
        n_layers=2,
        dropout=0.5,
        bidirectional=True
    )

    emb_cfg = EmbeddingConfig(model_name=embedding_model, max_len=embedding_max_len)

    return model, emb_cfg


def main():

    print("Starting BiLSTM model evaluation...")

    # data
    train_df = pd.read_csv("Input/opp115_train.csv")
    test_df = pd.read_csv("Input/opp115_test.csv")
    train_df, test_df = U.load_opp115_from_df(train_df, test_df)

    print(f"Loaded test data with {len(test_df)} samples.")

    texts = test_df["Sentence"].astype(str).tolist()
    y_true = test_df["label"].astype(int).to_numpy()

    # pick model
    model_dir = "artifacts/bilstm"
    meta, state = load_meta_and_state(model_dir)

    print(f"Loaded model meta: {meta}")

    model, emb_cfg = build_bilstm_from_meta(meta, state)
    model.load_state_dict(state)

    print("Model built from meta and state.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print(f"Model moved to device: {device}")

    print("Starting prediction...")

    with U.measure_run("EVAL - BiLSTM - OPP115"):
        y_pred_labels, _ = U.predict_bilstm(
            model=model,
            texts=texts,
            emb_cfg=emb_cfg,
            batch_size=256,
            return_proba=True,
        )

    print("Prediction completed.")

    # labels -> ids (meta moet mapping hebben, anders zit y_pred al als int)
    if isinstance(y_pred_labels[0], str):
        label2id = meta["labels"]  # expected dict name->id
        y_pred = np.array([label2id[lbl] for lbl in y_pred_labels], dtype=int)
    else:
        y_pred = np.asarray(y_pred_labels, dtype=int)

    metrics = compute_metrics(y_true, y_pred)

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out_df = test_df.copy()
    out_df["y_pred"] = y_pred
    out_df.to_csv(os.path.join(model_dir, "eval_predictions.csv"), index=False)

    print("Macro F1:", metrics["f1_macro"])
    print("Macro F2:", metrics["f2_macro"])
    print("Weighted F1:", metrics["f1_weighted"])
    print("Weighted F2:", metrics["f2_weighted"])


if __name__ == "__main__":
    main()
