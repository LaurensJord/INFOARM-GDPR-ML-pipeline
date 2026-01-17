#!/usr/bin/env python
# coding: utf-8
"""
Multi-seed training script for experimental validation.
Trains BERT, RoBERTa, and BiLSTM models with multiple random seeds
to demonstrate robustness and reproducibility of results.

Outputs structured data for statistical testing:
- Per-seed results (JSON)
- Summary statistics with confidence intervals
- CSV files ready for statistical analysis (t-tests, ANOVA, etc.)
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime
import Resources.utils_stripped as U

# Seeds for experimental validation
SEEDS = U.RANDOM_SEEDS  # [42, 123, 456]
METRICS = ["accuracy", "micro_f1", "micro_precision", "micro_recall", "macro_f1", "macro_precision", "macro_recall", "weighted_f1"]

def extract_full_metrics(test_result: dict) -> dict:
    """Extract all metrics from test result for statistical analysis."""
    report = test_result.get("report", {})
    
    metrics = {
        "accuracy": report.get("accuracy", 0),
        # Micro-averaged metrics (equivalent to accuracy for single-label classification)
        "micro_f1": report.get("micro avg", {}).get("f1-score", report.get("accuracy", 0)),
        "micro_precision": report.get("micro avg", {}).get("precision", report.get("accuracy", 0)),
        "micro_recall": report.get("micro avg", {}).get("recall", report.get("accuracy", 0)),
        # Macro-averaged metrics
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
        "macro_precision": report.get("macro avg", {}).get("precision", 0),
        "macro_recall": report.get("macro avg", {}).get("recall", 0),
        # Weighted-averaged metrics
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
        "weighted_precision": report.get("weighted avg", {}).get("precision", 0),
        "weighted_recall": report.get("weighted avg", {}).get("recall", 0),
    }
    
    # Per-class metrics
    for class_name in U.OPP115_CLASSES.keys():
        if class_name in report:
            metrics[f"{class_name}_f1"] = report[class_name].get("f1-score", 0)
            metrics[f"{class_name}_precision"] = report[class_name].get("precision", 0)
            metrics[f"{class_name}_recall"] = report[class_name].get("recall", 0)
            metrics[f"{class_name}_support"] = report[class_name].get("support", 0)
    
    return metrics


def train_all_models_with_seed(seed: int, df_train: pd.DataFrame, df_val: pd.DataFrame, test_df: pd.DataFrame):
    """Train all models with a specific seed and return results."""
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SEED: {seed}")
    print(f"{'='*60}\n")
    
    # Set the seed for reproducibility
    U.set_seed(seed)
    
    # Re-split train/val with new seed for this run
    df_train_seed, df_val_seed = U.split_train_val(df_train.copy(), val_size=0.2)
    
    results = {"seed": seed, "models": {}}
    
    # Train BiLSTM (first - fastest model)
    print(f"\n[Seed {seed}] Training BiLSTM...")
    U.set_seed(seed)  # Reset seed before each model
    with U.measure_run(f"TRAINING - BiLSTM - Seed {seed}"):
        bilstm_res = U.train_bilstm_mcc(
            df_train=df_train_seed,
            df_val=df_val_seed,
            epochs=3,
            batch_size=32,
            lr=1e-4,
            out_dir=f"artifacts/bilstm_seed_{seed}"
        )
    
    # Evaluate BiLSTM
    bilstm_test = U.evaluate_bilstm_mcc(
        model=bilstm_res["model"],
        df_test=test_df,
        batch_size=32
    )
    results["models"]["bilstm"] = {
        "best_val_macro_f1": bilstm_res["best_val_macro_f1"],
        "test_metrics": bilstm_test,
        "all_metrics": extract_full_metrics(bilstm_test)
    }
    print(f"[Seed {seed}] BiLSTM Test Macro F1: {bilstm_test.get('report', {}).get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
    
    # Train BERT
    print(f"\n[Seed {seed}] Training BERT...")
    U.set_seed(seed)  # Reset seed before each model
    with U.measure_run(f"TRAINING - BERT - Seed {seed}"):
        bert_res = U.train_transformer_mcc(
            model_type="bert",
            df_train=df_train_seed,
            df_val=df_val_seed,
            epochs=20,
            batch_size=64,
            lr=3e-5,
            out_dir=f"artifacts/bert_seed_{seed}"
        )
    
    # Evaluate BERT
    bert_test = U.evaluate_transformer_mcc(
        model=bert_res["model"],
        tokenizer=bert_res["tokenizer"],
        df_test=test_df,
        batch_size=64
    )
    results["models"]["bert"] = {
        "best_val_macro_f1": bert_res["best_val_macro_f1"],
        "test_metrics": bert_test,
        "all_metrics": extract_full_metrics(bert_test)
    }
    print(f"[Seed {seed}] BERT Test Macro F1: {bert_test.get('report', {}).get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
    
    # Train RoBERTa
    print(f"\n[Seed {seed}] Training RoBERTa...")
    U.set_seed(seed)  # Reset seed before each model
    with U.measure_run(f"TRAINING - RoBERTa - Seed {seed}"):
        roberta_res = U.train_transformer_mcc(
            model_type="roberta",
            df_train=df_train_seed,
            df_val=df_val_seed,
            epochs=20,
            batch_size=64,
            lr=3e-5,
            out_dir=f"artifacts/roberta_seed_{seed}"
        )
    
    # Evaluate RoBERTa
    roberta_test = U.evaluate_transformer_mcc(
        model=roberta_res["model"],
        tokenizer=roberta_res["tokenizer"],
        df_test=test_df,
        batch_size=64
    )
    results["models"]["roberta"] = {
        "best_val_macro_f1": roberta_res["best_val_macro_f1"],
        "test_metrics": roberta_test,
        "all_metrics": extract_full_metrics(roberta_test)
    }
    print(f"[Seed {seed}] RoBERTa Test Macro F1: {roberta_test.get('report', {}).get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
    
    return results


def compute_confidence_interval(values: list, confidence: float = 0.95) -> tuple:
    """Compute confidence interval for a list of values (numpy-only implementation)."""
    n = len(values)
    if n < 2:
        return (float(np.mean(values)), float(np.mean(values)))
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    # Approximate t-value for 95% CI with small samples (conservative)
    # For n=3, t_0.975,2 ≈ 4.303; for n=5, t_0.975,4 ≈ 2.776
    t_values = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042}
    df = n - 1
    t_val = t_values.get(df, 1.96)  # Default to z-value for large n
    h = se * t_val
    return (float(mean - h), float(mean + h))


def compute_summary_statistics(all_results: list) -> dict:
    """Compute comprehensive statistics across all seeds for each model."""
    summary = {}
    models = ["bert", "roberta", "bilstm"]
    
    # Collect all metrics for each model
    for model_name in models:
        model_metrics = {}
        
        for result in all_results:
            model_data = result["models"].get(model_name, {})
            all_metrics = model_data.get("all_metrics", {})
            
            for metric_name, value in all_metrics.items():
                if metric_name not in model_metrics:
                    model_metrics[metric_name] = []
                model_metrics[metric_name].append(value)
        
        # Compute statistics for each metric
        model_summary = {}
        for metric_name, values in model_metrics.items():
            if len(values) > 0:
                ci_low, ci_high = compute_confidence_interval(values)
                model_summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "ci_95_low": float(ci_low),
                    "ci_95_high": float(ci_high),
                    "values": [float(v) for v in values],
                    "n": len(values)
                }
        
        summary[model_name] = model_summary
    
    return summary


def create_statistical_dataframes(all_results: list, summary: dict) -> dict:
    """Create pandas DataFrames ready for statistical analysis."""
    models = ["bert", "roberta", "bilstm"]
    seeds = [r["seed"] for r in all_results]
    
    # Long-format DataFrame for repeated measures analysis
    long_data = []
    for result in all_results:
        seed = result["seed"]
        for model_name in models:
            model_data = result["models"].get(model_name, {})
            all_metrics = model_data.get("all_metrics", {})
            row = {"seed": seed, "model": model_name}
            row.update(all_metrics)
            long_data.append(row)
    
    df_long = pd.DataFrame(long_data)
    
    # Wide-format DataFrame (models as columns)
    wide_data = []
    for metric in METRICS:
        row = {"metric": metric}
        for model_name in models:
            if model_name in summary and metric in summary[model_name]:
                stats_data = summary[model_name][metric]
                row[f"{model_name}_mean"] = stats_data["mean"]
                row[f"{model_name}_std"] = stats_data["std"]
                row[f"{model_name}_ci95"] = f"[{stats_data['ci_95_low']:.4f}, {stats_data['ci_95_high']:.4f}]"
        wide_data.append(row)
    
    df_wide = pd.DataFrame(wide_data)
    
    # Per-seed comparison DataFrame
    comparison_data = []
    for result in all_results:
        seed = result["seed"]
        row = {"seed": seed}
        for model_name in models:
            model_data = result["models"].get(model_name, {})
            all_metrics = model_data.get("all_metrics", {})
            for metric in METRICS:
                row[f"{model_name}_{metric}"] = all_metrics.get(metric, np.nan)
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    return {
        "long_format": df_long,
        "wide_format": df_wide,
        "per_seed_comparison": df_comparison
    }


def print_latex_table(summary: dict):
    """Print a LaTeX-formatted table for academic papers."""
    models = ["bert", "roberta", "bilstm"]
    model_labels = {"bert": "BERT", "roberta": "RoBERTa", "bilstm": "BiLSTM"}
    
    print("\n" + "="*60)
    print("LATEX TABLE (for academic papers)")
    print("="*60)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Model Performance Comparison (Mean $\pm$ Std over 3 seeds)}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(r"Model & Accuracy & Micro F1 & Macro F1 & Macro P & Macro R & Weighted F1 \\")
    print(r"\hline")
    
    for model in models:
        if model not in summary:
            continue
        ms = summary[model]
        acc = ms.get("accuracy", {})
        mif1 = ms.get("micro_f1", {})
        maf1 = ms.get("macro_f1", {})
        mp = ms.get("macro_precision", {})
        mr = ms.get("macro_recall", {})
        wf1 = ms.get("weighted_f1", {})
        
        row = f"{model_labels[model]} & "
        row += f"{acc.get('mean', 0):.3f}$\\pm${acc.get('std', 0):.3f} & "
        row += f"{mif1.get('mean', 0):.3f}$\\pm${mif1.get('std', 0):.3f} & "
        row += f"{maf1.get('mean', 0):.3f}$\\pm${maf1.get('std', 0):.3f} & "
        row += f"{mp.get('mean', 0):.3f}$\\pm${mp.get('std', 0):.3f} & "
        row += f"{mr.get('mean', 0):.3f}$\\pm${mr.get('std', 0):.3f} & "
        row += f"{wf1.get('mean', 0):.3f}$\\pm${wf1.get('std', 0):.3f} \\\\"
        print(row)
    
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:model_comparison}")
    print(r"\end{table}")


def compute_summary_statistics(all_results: list) -> dict:
    """Compute mean and std across all seeds for each model."""
    summary = {}
    models = ["bert", "roberta", "bilstm"]
    
    # Collect all metrics for each model
    for model_name in models:
        model_metrics = {}
        
        for result in all_results:
            model_data = result["models"].get(model_name, {})
            all_metrics = model_data.get("all_metrics", {})
            
            for metric_name, value in all_metrics.items():
                if metric_name not in model_metrics:
                    model_metrics[metric_name] = []
                model_metrics[metric_name].append(value)
        
        # Compute statistics for each metric
        model_summary = {}
        for metric_name, values in model_metrics.items():
            if len(values) > 0:
                ci_low, ci_high = compute_confidence_interval(values)
                model_summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "ci_95_low": float(ci_low),
                    "ci_95_high": float(ci_high),
                    "values": [float(v) for v in values],
                    "n": len(values)
                }
        
        summary[model_name] = model_summary
    
    return summary


def main():
    print("="*60)
    print("MULTI-SEED TRAINING FOR EXPERIMENTAL VALIDATION")
    print(f"Seeds: {SEEDS}")
    print("="*60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_df = pd.read_csv("Input/opp115_train.csv")
    test_df = pd.read_csv("Input/opp115_test.csv")
    train_df, test_df = U.load_opp115_from_df(train_df, test_df)
    
    # Train with each seed
    all_results = []
    for seed in SEEDS:
        result = train_all_models_with_seed(seed, train_df, train_df, test_df)
        all_results.append(result)
        
        # Save individual seed results
        os.makedirs("artifacts/multi_seed", exist_ok=True)
        with open(f"artifacts/multi_seed/results_seed_{seed}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    
    # Compute summary statistics
    summary = compute_summary_statistics(all_results)
    summary["seeds"] = SEEDS
    summary["timestamp"] = datetime.now().isoformat()
    
    # Create DataFrames for statistical analysis
    dfs = create_statistical_dataframes(all_results, summary)
    
    # Save all outputs
    output_dir = "artifacts/multi_seed"
    
    # Save summary JSON
    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save CSVs for statistical analysis
    dfs["long_format"].to_csv(f"{output_dir}/results_long_format.csv", index=False)
    dfs["wide_format"].to_csv(f"{output_dir}/results_summary_wide.csv", index=False)
    dfs["per_seed_comparison"].to_csv(f"{output_dir}/results_per_seed.csv", index=False)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("MULTI-SEED TRAINING SUMMARY")
    print("="*60)
    print(f"\nSeeds used: {SEEDS}")
    
    print("\n" + "-"*60)
    print("RESULTS (Mean ± Std | 95% CI)")
    print("-"*60)
    
    models = ["bert", "roberta", "bilstm"]
    for model_name in models:
        print(f"\n{model_name.upper()}:")
        if model_name in summary:
            for metric in METRICS:
                if metric in summary[model_name]:
                    s = summary[model_name][metric]
                    print(f"  {metric:20s}: {s['mean']:.4f} ± {s['std']:.4f}  [{s['ci_95_low']:.4f}, {s['ci_95_high']:.4f}]")
    
    # Print LaTeX table
    print_latex_table(summary)
    
    # Print output file locations
    print("\n" + "="*60)
    print("OUTPUT FILES (ready for external statistical analysis)")
    print("="*60)
    print(f"\n  {output_dir}/summary.json              - Summary statistics")
    print(f"  {output_dir}/results_long_format.csv   - Long format for ANOVA/mixed models")
    print(f"  {output_dir}/results_summary_wide.csv  - Wide format summary table")
    print(f"  {output_dir}/results_per_seed.csv      - Per-seed comparison")
    print(f"\nIndividual seed results:")
    for seed in SEEDS:
        print(f"  {output_dir}/results_seed_{seed}.json")


if __name__ == "__main__":
    main()
