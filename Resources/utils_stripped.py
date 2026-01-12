# Minimal OPP-115 utils (multi-class) for BERT, RoBERTa, BiLSTM
# Stripped from original pipeline.

from __future__ import annotations

import os, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import psutil
from contextlib import contextmanager

from transformers import (
    BertModel, BertTokenizerFast,
    RobertaModel, RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sentence_transformers import SentenceTransformer

# Default seed for single-run experiments
RANDOM_SEED = 42
# Multiple seeds for experimental validation (to demonstrate robustness)
RANDOM_SEEDS = [42, 123, 456]

def set_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Initialize with default seed
set_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 160
DROPOUT = 0.3

# OPP-115 Privacy Policy Dataset - 9 classes
OPP115_CLASSES: Dict[str, int] = {
    "DataCollection": 0,
    "ThirdPartySharing": 1,
    "UserRights": 2,
    "DataRetention": 3,
    "DataSecurity": 4,
    "PolicyChange": 5,
    "DoNotTrack": 6,
    "SpecialAudiences": 7,
    "Other": 8,
}
ID_TO_CLASS = {v: k for k, v in OPP115_CLASSES.items()}
N_CLASSES = len(OPP115_CLASSES)
LABEL_IDS = list(range(N_CLASSES))
LABEL_NAMES = [ID_TO_CLASS[i] for i in range(N_CLASSES)]


def ensure_opp115_labels(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Sentence", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["target"] = out["target"].astype(str)

    unknown = sorted(set(out["target"].unique()) - set(OPP115_CLASSES.keys()))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Expected: {sorted(OPP115_CLASSES.keys())}")

    if "label" not in out.columns:
        out["label"] = out["target"].map(OPP115_CLASSES).astype(int)
    else:
        out["label"] = out["label"].astype(int)
        mismatch = out[out["label"] != out["target"].map(OPP115_CLASSES)]
        if len(mismatch) > 0:
            raise ValueError("label column does not match OPP115_CLASSES mapping for some rows.")
    return out


def load_opp115_from_df(df_train: pd.DataFrame, df_test) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df_train = ensure_opp115_labels(df_train)
    df_test = ensure_opp115_labels(df_test)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def split_train_val(train_df: pd.DataFrame, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = ensure_opp115_labels(train_df)
    tr, va = train_test_split(
        train_df, test_size=val_size, random_state=RANDOM_SEED, stratify=train_df["label"]
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


class TextClsDataset(Dataset):
    def __init__(self, sentences: np.ndarray, labels: np.ndarray, tokenizer, max_len: int):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        text = str(self.sentences[idx])
        y = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "sentence_text": text,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": torch.tensor(y, dtype=torch.long),
        }


def make_text_dataloader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    max_len: int = MAX_LEN,
    class_weighted_sampler: bool = True,
) -> DataLoader:
    df = ensure_opp115_labels(df)
    ds = TextClsDataset(
        sentences=df["Sentence"].to_numpy(),
        labels=df["label"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    if not class_weighted_sampler:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    counts = df["label"].value_counts().sort_index()
    weights_by_class = (counts.sum() / counts).to_dict()
    sample_weights = df["label"].map(weights_by_class).astype(float).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=2)


class BertClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.backbone = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=DROPOUT)
        self.out = nn.Linear(self.backbone.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(pooled))


class RobertaClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, model_name: str = "roberta-base"):
        super().__init__()
        self.backbone = RobertaModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=DROPOUT)
        self.out = nn.Linear(self.backbone.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.out(self.drop(pooled))


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_len: int = 160


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.x = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        output_dim: int = N_CLASSES,
        n_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(lstm_out, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_embed):
        x = x_embed.unsqueeze(1)  # [B,1,D]
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            cat = hidden[-1, :, :]
        x = self.relu(cat)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


def embed_sentences(df: pd.DataFrame, cfg: EmbeddingConfig) -> np.ndarray:
    df = ensure_opp115_labels(df)
    st = SentenceTransformer(cfg.model_name, device=str(DEVICE))
    st.max_seq_length = cfg.max_len
    return st.encode(df["Sentence"].tolist(), convert_to_numpy=True, show_progress_bar=True)


def _epoch_transformer_train(model, loader, loss_fn, optimizer, scheduler) -> float:
    model.train()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention = batch["attention_mask"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention)
        loss = loss_fn(logits, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _epoch_transformer_eval(model, loader, loss_fn) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, preds_all, y_all = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention = batch["attention_mask"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)

        logits = model(input_ids=input_ids, attention_mask=attention)
        loss = loss_fn(logits, targets)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        preds_all.append(preds.detach().cpu().numpy())
        y_all.append(targets.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all) if preds_all else np.array([])
    y_all = np.concatenate(y_all) if y_all else np.array([])
    return (float(np.mean(losses)) if losses else 0.0, preds_all, y_all)


def train_transformer_mcc(
    model_type: Literal["bert", "roberta"],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    out_dir: Optional[str] = None,
) -> Dict:
    print(f"A. entered train_transformer_mcc (model_type={model_type})")
    df_train = ensure_opp115_labels(df_train)
    df_val = ensure_opp115_labels(df_val)
    print("B. labels ensured")

    if model_type == "bert":
        backbone = "bert-base-uncased"
        print(f"C. loading BERT tokenizer from {backbone}")
        tokenizer = BertTokenizerFast.from_pretrained(backbone)
        print("D. tokenizer loaded, loading BERT model")
        model = BertClassifier(N_CLASSES, backbone)
        print("E. BERT model created")
    elif model_type == "roberta":
        backbone = "roberta-base"
        print(f"C. loading RoBERTa tokenizer from {backbone}")
        tokenizer = RobertaTokenizerFast.from_pretrained(backbone)
        print("D. tokenizer loaded, loading RoBERTa model")
        model = RobertaClassifier(N_CLASSES, backbone)
        print("E. RoBERTa model created")
    else:
        raise ValueError("model_type must be 'bert' or 'roberta'")

    print("F. moving model to device")
    model.to(DEVICE)
    print(f"G. model on {DEVICE}")

    print("H. creating train dataloader")
    train_loader = make_text_dataloader(df_train, tokenizer, batch_size, class_weighted_sampler=True)
    print(f"I. train dataloader created ({len(train_loader)} batches)")
    val_loader = make_text_dataloader(df_val, tokenizer, batch_size, class_weighted_sampler=False)
    print(f"J. val dataloader created ({len(val_loader)} batches)")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    best_macro_f1 = -1.0
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        tr_loss = _epoch_transformer_train(model, train_loader, loss_fn, optimizer, scheduler)
        va_loss, va_pred, va_y = _epoch_transformer_eval(model, val_loader, loss_fn)

        report = classification_report(
            va_y, va_pred, labels=LABEL_IDS, target_names=LABEL_NAMES, output_dict=True, zero_division=0
        )
        macro_f1 = float(report["macro avg"]["f1-score"])
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "val_macro_f1": macro_f1})

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"{model_type}_state.bin"))
        with open(os.path.join(out_dir, f"{model_type}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"model_type": model_type, "n_classes": N_CLASSES, "labels": OPP115_CLASSES, "max_len": MAX_LEN}, f, indent=2)

    return {"model": model, "tokenizer": tokenizer, "history": history, "best_val_macro_f1": best_macro_f1}


@torch.no_grad()
def predict_transformer(model, tokenizer, texts: List[str], batch_size: int = 64, return_proba: bool = False):
    model.eval()
    all_probs, all_pred = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch, add_special_tokens=True, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors="pt",
        )
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        all_probs.append(probs)
        all_pred.extend([ID_TO_CLASS[int(j)] for j in pred_ids])
    proba = np.vstack(all_probs) if return_proba else None
    return all_pred, proba


def train_bilstm_mcc(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    emb_cfg: EmbeddingConfig = EmbeddingConfig(),
    hidden_dim: int = 768,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-4,
    out_dir: Optional[str] = None,
) -> Dict:
    print("A. entered train_bilstm_mcc")
    df_train = ensure_opp115_labels(df_train)
    df_val = ensure_opp115_labels(df_val)
    print("B. labels ensured")

    Xtr = embed_sentences(df_train, emb_cfg)
    print(f"C. training embeddings done: {Xtr.shape}")
    Xva = embed_sentences(df_val, emb_cfg)
    print(f"D. validation embeddings done: {Xva.shape}")
    input_size = int(Xtr.shape[1])

    model = BiLSTMClassifier(input_size=input_size, hidden_dim=hidden_dim, output_dim=N_CLASSES).to(DEVICE)
    counts = df_train["label"].value_counts().sort_index()
    weights_by_class = (counts.sum() / counts).to_dict()
    sample_weights = df_train["label"].map(weights_by_class).astype(float).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(EmbeddingDataset(Xtr, df_train["label"].to_numpy()), batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(EmbeddingDataset(Xva, df_val["label"].to_numpy()), batch_size=batch_size, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    best_macro_f1 = -1.0
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses, preds, ys = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                va_losses.append(loss_fn(logits, yb).item())
                preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
                ys.append(yb.detach().cpu().numpy())

        va_pred = np.concatenate(preds) if preds else np.array([])
        va_y = np.concatenate(ys) if ys else np.array([])
        report = classification_report(
            va_y, va_pred, labels=LABEL_IDS, target_names=LABEL_NAMES, output_dict=True, zero_division=0
        )
        macro_f1 = float(report["macro avg"]["f1-score"])
        history.append({"epoch": ep, "train_loss": float(np.mean(tr_losses)) if tr_losses else 0.0,
                        "val_loss": float(np.mean(va_losses)) if va_losses else 0.0,
                        "val_macro_f1": macro_f1})

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "bilstm_state.bin"))
        with open(os.path.join(out_dir, "bilstm_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"n_classes": N_CLASSES, "labels": OPP115_CLASSES,
                       "embedding_model": emb_cfg.model_name, "embedding_max_len": emb_cfg.max_len,
                       "input_size": input_size}, f, indent=2)

    return {"model": model, "embedding_config": emb_cfg, "history": history, "best_val_macro_f1": best_macro_f1}


@torch.no_grad()
def predict_bilstm(model, texts: List[str], emb_cfg: EmbeddingConfig = EmbeddingConfig(), batch_size: int = 256, return_proba: bool = False):
    model.eval()
    st = SentenceTransformer(emb_cfg.model_name, device=str(DEVICE))
    st.max_seq_length = emb_cfg.max_len

    all_probs, all_pred = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        X = st.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        all_probs.append(probs)
        all_pred.extend([ID_TO_CLASS[int(j)] for j in pred_ids])
    proba = np.vstack(all_probs) if return_proba else None
    return all_pred, proba


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    report = classification_report(y_true, y_pred, labels=LABEL_IDS, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    return {"report": report, "confusion_matrix": cm, "label_names": LABEL_NAMES}

@contextmanager
def measure_run(tag="run"):
    proc = psutil.Process(os.getpid())

    # CPU / RAM (start)
    rss_start = proc.memory_info().rss
    cpu_start = proc.cpu_times()
    t0 = time.perf_counter()

    # GPU (start)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    yield  # ---- run training or inference here ----

    # GPU (end)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        gpu_peak_mb = None

    # CPU / RAM (end)
    t1 = time.perf_counter()
    rss_end = proc.memory_info().rss
    cpu_end = proc.cpu_times()

    result = {
        "tag": tag,
        "wall_time_sec": t1 - t0,
        "cpu_user_sec": cpu_end.user - cpu_start.user,
        "cpu_system_sec": cpu_end.system - cpu_start.system,
        "ram_delta_mb": (rss_end - rss_start) / (1024**2),
        "gpu_peak_mem_mb": gpu_peak_mb,
    }

    print("=== PERF ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Save to file - use path relative to this script's parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_dir = os.path.join(os.path.dirname(script_dir), "artifacts", "performance")
    os.makedirs(perf_dir, exist_ok=True)
    
    perf_file = os.path.join(perf_dir, f"perf_{tag}.json")
    with open(perf_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    
    return result
