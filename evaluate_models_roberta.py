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

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def fbeta_from_pr(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


@torch.no_grad()
def predict_transformer(model, tokenizer, texts, batch_size=64, max_length=256, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        logits = out.logits
        batch_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        preds.append(batch_pred)

    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # Macro / weighted / micro for P/R/F1
    metrics = {}
    for avg in ["macro", "weighted", "micro"]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = float(p)
        metrics[f"recall_{avg}"] = float(r)
        metrics[f"f1_{avg}"] = float(f1)
        metrics[f"f2_{avg}"] = float(fbeta_from_pr(float(p), float(r), beta=2.0))

    # Per-class report + confusion matrix
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics["classification_report"] = report
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def main():

    print("Starting RoBERTa model evaluation...")

    # 1) Load test data and normalize exactly like training
    test_df = pd.read_csv("Input/opp115_test.csv")
    # If your loader requires both train+test, load train too (keeps label mapping consistent)
    train_df = pd.read_csv("Input/opp115_train.csv")
    train_df, test_df = U.load_opp115_from_df(train_df, test_df)

    # Adjust these column names to your actual schema after load_opp115_from_df
    texts = test_df["Sentence"].astype(str).tolist()
    y_true = test_df["label"].astype(int).to_numpy()

    # 2) Load saved model/tokenizer
    model_dir = "artifacts/artifacts/roberta"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Predict
    with U.measure_run("EVAL - RoBERTa - OPP115"):
        y_pred = predict_transformer(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=64,
            max_length=256,
            device=device,
        )

    # 4) Metrics
    metrics = compute_metrics(y_true, y_pred)

    # 5) Save outputs
    os.makedirs(model_dir, exist_ok=True)

    metrics_path = os.path.join(model_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_path = os.path.join(model_dir, "eval_predictions.csv")
    out_df = test_df.copy()
    out_df["y_pred"] = y_pred
    out_df.to_csv(pred_path, index=False)

    # 6) Minimal console output
    print("Saved:", metrics_path)
    print("Saved:", pred_path)
    print("Macro F1:", metrics["f1_macro"])
    print("Macro F2:", metrics["f2_macro"])
    print("Weighted F1:", metrics["f1_weighted"])
    print("Weighted F2:", metrics["f2_weighted"])


if __name__ == "__main__":
    main()
