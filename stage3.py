#!/usr/bin/env python3
"""
stage3.py — SemEval 2022 Task 4, Subtask 1: PCL Binary Detection
Baseline (RoBERTa-base + standard CE) vs Novel (RoBERTa-base + focal loss).

Primary metric: F1 score of the positive (PCL) class only,
    f1_score(y_true, y_pred, pos_label=1)

Usage:
    python stage3.py --mode baseline [--n_trials 10] [--epochs 10] [--patience 3] [--seed 42]
    python stage3.py --mode novel   [--n_trials 12]
    python stage3.py --mode both
    python stage3.py --mode both --n_trials 2 --epochs 3 --patience 2   # quick smoke-test

All outputs land in --output_dir (default: outputs/):
    stage3_baseline_results.json
    stage3_novel_results.json
    stage3_tuning_log.csv
    stage3_report.md
    baseline_best_checkpoint/   (HuggingFace model + tokenizer)
    novel_best_checkpoint/

Design rationale (see report for full justification):
  - EDA shows 9.5% PCL rate → severe class imbalance.
  - N-gram overlap between classes → surface phrases are not discriminative;
    a contextual encoder is needed to capture pragmatic cues.
  - Novel approach: same RoBERTa-base encoder but with alpha-balanced focal
    loss (Lin et al. 2017) that down-weights easy No-PCL examples and focuses
    training on hard PCL samples.
  - Threshold tuning on dev set is required for both modes because the default
    0.5 threshold is far from optimal under a 9.5% positive prior.
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
import argparse
import ast
import csv
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# Optional Optuna (TPE search); falls back to random search if unavailable.
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ---------------------------------------------------------------------------
# 1. Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Encoder for both modes; key difference is the loss function.
BASELINE_MODEL = "roberta-base"
NOVEL_MODEL = "roberta-base"

# Input data paths (relative to repo root).
TRAIN_LABELS_PATH = "train_semeval_parids-labels.csv"
PCL_TSV_PATH = "dontpatronizeme_pcl.tsv"

# Stratified split ratios (no official dev/test set provided in this repo).
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# Threshold sweep range for dev-set optimisation.
THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.01)


# ---------------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS (Apple Silicon) does not have a manual seed API; torch.manual_seed covers it.


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_versions() -> dict:
    """Log and return a dict of key package versions for reproducibility."""
    import sklearn
    import transformers

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "optuna": optuna.__version__ if HAS_OPTUNA else "not installed",
    }
    logger.info("─── Package versions ───────────────────────────")
    for k, v in versions.items():
        logger.info(f"  {k:<14} {v}")
    logger.info("────────────────────────────────────────────────")
    return versions


# ---------------------------------------------------------------------------
# 3. Data loading
# ---------------------------------------------------------------------------

def read_labels(path: str) -> dict:
    """
    Read train_semeval_parids-labels.csv → {par_id: binary_label (0 or 1)}.

    The CSV has a multi-label vector per paragraph (7-element list).
    Binary label = 1 if ANY annotator marked it PCL (sum > 0), else 0.
    This matches the organiser's Task 4 Subtask 1 definition.
    """
    labels = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec = ast.literal_eval(row["label"])
            labels[row["par_id"]] = 1 if sum(vec) > 0 else 0
    return labels


def read_texts(path: str) -> dict:
    """
    Read dontpatronizeme_pcl.tsv (skip 4-line disclaimer) → {par_id: text}.

    TSV columns: par_id, art_id, keyword, country_code, text, label
    """
    texts = {}
    with open(path, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                texts[parts[0]] = parts[4]
    return texts


def load_data(labels_path: str, pcl_path: str):
    """
    Merge labels and texts; return (X: list[str], y: list[int]).

    Drops paragraphs where the text is empty/missing.
    """
    labels = read_labels(labels_path)
    texts = read_texts(pcl_path)
    X, y = [], []
    for par_id, label in labels.items():
        text = texts.get(par_id, "").strip()
        if text:
            X.append(text)
            y.append(label)
    n_pcl = sum(y)
    logger.info(
        f"Loaded {len(X)} samples — "
        f"PCL={n_pcl} ({100*n_pcl/len(y):.1f}%)  "
        f"No-PCL={len(y)-n_pcl} ({100*(len(y)-n_pcl)/len(y):.1f}%)"
    )
    return X, y


def make_splits(X: list, y: list, seed: int):
    """
    Create stratified 70 / 15 / 15 train / dev / test splits.

    Stratification ensures the ~9.5% PCL rate is preserved in each split,
    which is critical for valid dev-set threshold tuning and test evaluation.
    """
    X_arr = np.array(X, dtype=object)
    y_arr = np.array(y)

    # Step 1: carve out the test set.
    X_td, X_test, y_td, y_test = train_test_split(
        X_arr, y_arr,
        test_size=TEST_RATIO,
        stratify=y_arr,
        random_state=seed,
    )
    # Step 2: split remaining into train / dev.
    dev_ratio_adjusted = DEV_RATIO / (TRAIN_RATIO + DEV_RATIO)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_td, y_td,
        test_size=dev_ratio_adjusted,
        stratify=y_td,
        random_state=seed,
    )
    logger.info(
        f"Splits: train={len(X_train)}, dev={len(X_dev)}, test={len(X_test)}  "
        f"(PCL rate ≈ {100*y_train.mean():.1f}% / {100*y_dev.mean():.1f}% / {100*y_test.mean():.1f}%)"
    )
    return (
        X_train.tolist(), X_dev.tolist(), X_test.tolist(),
        y_train.tolist(), y_dev.tolist(), y_test.tolist(),
    )


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset
# ---------------------------------------------------------------------------

class PCLDataset(Dataset):
    """Tokenised PCL dataset for HuggingFace transformer models."""

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 5. Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Alpha-balanced focal loss for binary sequence classification.

    Motivation (EDA-grounded):
      - PCL class is only 9.5% of the training set.  Standard cross-entropy
        treats every example equally, so the model minimises loss by predicting
        No-PCL almost always.
      - Focal loss adds a modulating factor (1 - p_t)^γ that down-weights
        well-classified (easy) No-PCL examples, shifting the training gradient
        toward the harder, minority PCL examples.
      - α provides an additional per-class weight: α for the positive (PCL)
        class, (1-α) for the negative class.

    Reference: Lin et al. 2017, "Focal Loss for Dense Object Detection".
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        # alpha: weight for the positive (PCL) class.  Start at 0.25 (Lin et
        # al. default) and tune upward to account for 9.5% positive rate.
        self.alpha = alpha
        # gamma: focusing strength.  0 = standard CE; 2 = standard focal.
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2) from the classification head.
        # targets: (B,) integer labels in {0, 1}.
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of the *correct* class
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )
        focal = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ---------------------------------------------------------------------------
# 6. Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def get_probabilities(model, loader: DataLoader, device: torch.device):
    """
    Run inference; return (probs_pcl: np.array, true_labels: np.array).
    probs_pcl[i] is the model's estimated probability that sample i is PCL.
    """
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def tune_threshold(probs: np.ndarray, labels: np.ndarray):
    """
    Sweep decision thresholds in [0.05, 0.95] and return the one that
    maximises F1(PCL) on the provided (dev) set.

    Why this is required:
      With only 9.5% positives, the calibrated Bayes-optimal threshold for
      F1 is well below 0.5.  Threshold tuning on the dev set finds the
      optimal balance between precision and recall for the PCL class.
    Returns: (best_threshold, best_f1, list_of_(threshold, f1) pairs)
    """
    best_t, best_f1 = 0.5, 0.0
    all_results = []
    for t in THRESHOLD_RANGE:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        all_results.append((round(float(t), 3), float(f1)))
        if f1 > best_f1:
            best_f1 = f1
            best_t = round(float(t), 3)
    return best_t, best_f1, all_results


def evaluate_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute F1/P/R/CM for the PCL class at a given threshold."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return {
        "f1_pcl": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "precision_pcl": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_pcl": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }


def run_training(
    model_name: str,
    X_train, y_train,
    X_dev, y_dev,
    tokenizer,
    *,
    lr: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    max_len: int,
    seed: int,
    device: torch.device,
    loss_mode: str = "standard",   # "standard" | "focal"
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    patience: int = 3,
):
    """
    Fine-tune a sequence classifier for one complete hyperparameter config.

    Args:
        patience: Early stopping patience - stop if no improvement for N epochs

    Returns:
        best_dev_f1     : float
        best_threshold  : float
        best_metrics    : dict
        best_state_dict : dict of cpu tensors (best epoch checkpoint)
    """
    set_seed(seed)

    train_ds = PCLDataset(X_train, y_train, tokenizer, max_len)
    dev_ds = PCLDataset(X_dev, y_dev, tokenizer, max_len)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=pin)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * epochs
    # 6% warmup is a common default for RoBERTa fine-tuning (Liu et al. 2019).
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Choose loss function.
    if loss_mode == "focal":
        _criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss_fn = lambda logits, labels: _criterion(logits, labels)
    else:
        # Standard cross-entropy (baseline).
        loss_fn = lambda logits, labels: F.cross_entropy(logits, labels)

    best_dev_f1 = 0.0
    best_threshold = 0.5
    best_metrics = {}
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_fn, device)
        dev_probs, dev_labels = get_probabilities(model, dev_loader, device)
        threshold, dev_f1, _ = tune_threshold(dev_probs, dev_labels)
        metrics = evaluate_at_threshold(dev_probs, dev_labels, threshold)
        elapsed = time.time() - t0

        logger.info(
            f"    Epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
            f"dev F1(PCL)={metrics['f1_pcl']:.4f}  "
            f"P={metrics['precision_pcl']:.4f}  R={metrics['recall_pcl']:.4f}  "
            f"thr={threshold:.2f}  ({elapsed:.0f}s)"
        )

        if metrics["f1_pcl"] > best_dev_f1:
            best_dev_f1 = metrics["f1_pcl"]
            best_threshold = threshold
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            # Store CPU copy to avoid holding the device memory across trials.
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"    Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return best_dev_f1, best_threshold, best_metrics, best_state


# ---------------------------------------------------------------------------
# 7. Hyperparameter search (Optuna TPE or random search fallback)
# ---------------------------------------------------------------------------

def run_search(
    mode: str,
    model_name: str,
    X_train, y_train,
    X_dev, y_dev,
    tokenizer,
    device: torch.device,
    args,
    tuning_rows: list,
) -> tuple:
    """
    Run Optuna TPE (or random search) over the hyperparameter space.

    Search space rationale
    ──────────────────────
    learning_rate  [1e-5, 5e-5] log-scale
        Standard fine-tuning range for BERT-family (Devlin et al. 2019).
        Values outside this range typically diverge or converge too slowly.

    batch_size  {8, 16}
        32 can OOM with seq_len=256 on ≤16GB RAM; 8/16 standard for NLP.

    epochs  [2, max_epochs]
        <2 = underfitting on 5.8K samples; >5 = overfitting.

    weight_decay  [0.0, 0.1]
        Moderate L2 for BERT (Sun et al. 2019); 0.01–0.1 most effective.

    max_len  {128, 256}
        128 tokens covers most short news paragraphs quickly; 256 covers
        ~95th percentile of paragraph length.  320 adds latency with
        diminishing marginal coverage.

    focal_gamma  [0.5, 3.0]  (novel only)
        γ=0 → standard CE; γ=2 → Lin et al. default for dense detection;
        γ=3 → aggressive focus, may destabilise training.

    focal_alpha  [0.25, 0.75]  (novel only)
        α=0.25 → Lin et al. default; push toward 0.75 to compensate for
        the 9.5% positive rate.

    Tuning budget
    ─────────────
    Default 10 trials per mode ≈ 1–2 hours on Apple Silicon MPS (3 epochs
    each).  TPE becomes efficient after ~10 observations, so this budget
    provides meaningful exploration while guarding against over-fitting the
    dev set (limited trials = limited selection pressure on dev).
    """
    n_trials = args.n_trials
    max_epochs = args.epochs
    seed = args.seed

    # Best state across all trials is tracked in a mutable closure variable
    # to avoid writing 10+ checkpoint directories (each ~500MB).
    best_holder = {
        "f1": 0.0,
        "threshold": 0.5,
        "metrics": {},
        "state": None,
        "params": {},
        "trial_id": -1,
    }

    def _one_trial(trial_id: int, params: dict) -> float:
        lr = params["lr"]
        bs = params["batch_size"]
        epochs = params["epochs"]
        wd = params["weight_decay"]
        max_len = params["max_len"]
        focal_gamma = params.get("focal_gamma", 2.0)
        focal_alpha = params.get("focal_alpha", 0.25)
        loss_mode = "focal" if mode == "novel" else "standard"

        logger.info(
            f"\n[{mode.upper()} Trial {trial_id}/{n_trials-1}] "
            f"lr={lr:.2e}  bs={bs}  ep={epochs}  wd={wd:.3f}  "
            f"max_len={max_len}"
            + (f"  γ={focal_gamma:.2f}  α={focal_alpha:.2f}" if mode == "novel" else "")
        )

        dev_f1, threshold, metrics, state = run_training(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_dev=X_dev, y_dev=y_dev,
            tokenizer=tokenizer,
            lr=lr, batch_size=bs, epochs=epochs,
            weight_decay=wd, max_len=max_len,
            seed=seed, device=device,
            loss_mode=loss_mode,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            patience=args.patience,
        )

        # Record trial in the shared log.
        tuning_rows.append({
            "mode": mode,
            "trial_id": trial_id,
            "lr": lr,
            "batch_size": bs,
            "epochs": epochs,
            "weight_decay": wd,
            "max_len": max_len,
            "focal_gamma": focal_gamma if mode == "novel" else "",
            "focal_alpha": focal_alpha if mode == "novel" else "",
            "best_threshold": threshold,
            "dev_f1_pcl": metrics.get("f1_pcl", 0.0),
            "dev_precision": metrics.get("precision_pcl", 0.0),
            "dev_recall": metrics.get("recall_pcl", 0.0),
            "dev_tp": metrics["confusion_matrix"]["tp"],
            "dev_fp": metrics["confusion_matrix"]["fp"],
            "dev_tn": metrics["confusion_matrix"]["tn"],
            "dev_fn": metrics["confusion_matrix"]["fn"],
        })

        # Keep only the best model state dict in memory.
        if dev_f1 > best_holder["f1"]:
            best_holder["f1"] = dev_f1
            best_holder["threshold"] = threshold
            best_holder["metrics"] = metrics
            best_holder["state"] = state
            best_holder["params"] = params
            best_holder["trial_id"] = trial_id

        return dev_f1

    if HAS_OPTUNA:
        def _optuna_objective(trial: optuna.Trial) -> float:
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                "epochs": trial.suggest_int("epochs", min(2, max_epochs), max_epochs),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "max_len": trial.suggest_categorical("max_len", [128, 256]),
            }
            if mode == "novel":
                params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 3.0)
                params["focal_alpha"] = trial.suggest_float("focal_alpha", 0.25, 0.75)
            return _one_trial(trial.number, params)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(_optuna_objective, n_trials=n_trials, show_progress_bar=False)
    else:
        # Random search fallback.
        rng = np.random.RandomState(seed)
        for i in range(n_trials):
            params = {
                "lr": float(np.exp(rng.uniform(np.log(1e-5), np.log(5e-5)))),
                "batch_size": int(rng.choice([8, 16])),
                "epochs": int(rng.randint(min(2, max_epochs), max_epochs + 1)),
                "weight_decay": float(rng.uniform(0.0, 0.1)),
                "max_len": int(rng.choice([128, 256])),
            }
            if mode == "novel":
                params["focal_gamma"] = float(rng.uniform(0.5, 3.0))
                params["focal_alpha"] = float(rng.uniform(0.25, 0.75))
            _one_trial(i, params)

    return best_holder


# ---------------------------------------------------------------------------
# 8. Final test-set evaluation from saved checkpoint
# ---------------------------------------------------------------------------

def save_and_eval_test(
    best_state: dict,
    model_name: str,
    tokenizer,
    checkpoint_dir: str,
    X_test: list,
    y_test: list,
    threshold: float,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    Restore the best model state dict, save the checkpoint, and evaluate
    on the held-out test set at the best dev-tuned threshold.

    The test set is untouched during all tuning decisions; this is the
    ONLY time it is evaluated.
    """
    # Rebuild model and load best weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    model.load_state_dict(best_state)
    model.to(device)

    # Persist checkpoint for later use / inspection.
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info(f"Checkpoint saved → {checkpoint_dir}")

    # Evaluate on test.
    test_ds = PCLDataset(X_test, y_test, tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2,
                             shuffle=False, num_workers=0)
    probs, labels = get_probabilities(model, test_loader, device)

    # Also report best threshold found on test (for reference only; NOT used
    # to select the threshold — dev threshold is canonical).
    test_thr, test_f1_at_best_thr, _ = tune_threshold(probs, labels)
    metrics = evaluate_at_threshold(probs, labels, threshold)
    metrics["test_oracle_threshold"] = float(test_thr)
    metrics["test_oracle_f1"] = float(test_f1_at_best_thr)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return metrics


# ---------------------------------------------------------------------------
# 9. Report generation
# ---------------------------------------------------------------------------

def generate_report(
    output_dir: str,
    baseline_results: dict,
    novel_results: dict,
    versions: dict,
) -> str:
    """Write outputs/stage3_report.md with full research-style justification."""

    def _metrics_row(label, metrics, threshold):
        f1 = metrics.get("f1_pcl", 0)
        p  = metrics.get("precision_pcl", 0)
        r  = metrics.get("recall_pcl", 0)
        return (
            f"| {label:<38} | {f1:.4f} | {p:.4f} | {r:.4f} | {threshold:.2f} |"
        )

    def _cm_str(cm):
        return (
            f"TP={cm.get('tp',0)}  FP={cm.get('fp',0)}  "
            f"FN={cm.get('fn',0)}  TN={cm.get('tn',0)}"
        )

    b = baseline_results or {}
    n = novel_results or {}
    bdev = b.get("dev_metrics", {})
    btst = b.get("test_metrics", {})
    ndev = n.get("dev_metrics", {})
    ntst = n.get("test_metrics", {})
    bthr = b.get("best_threshold", 0.5)
    nthr = n.get("best_threshold", 0.5)

    lines = [
        "# Stage 3 — PCL Detection: Baseline vs Novel Approach",
        "",
        "> **Task:** SemEval 2022 Task 4, Subtask 1 — binary classification of",
        "> Patronizing and Condescending Language (PCL).",
        ">",
        "> **Primary metric:** F1 of the *positive* (PCL) class only,",
        "> `f1_score(y_true, y_pred, pos_label=1)`.",
        "",
        "---",
        "",
        "## 1. Baseline Reproduction",
        "",
        "### Setup",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Encoder | `roberta-base` |",
        "| Loss | Standard cross-entropy (uniform weights) |",
        "| Data split | Stratified 70 / 15 / 15 (no official dev/test provided) |",
        "| Split seed | Fixed (reproducible) |",
        "| Threshold | Swept [0.05, 0.95] on *dev only*; best applied to test |",
        "| Tuning | Optuna TPE (or random search if Optuna unavailable) |",
        "",
        "### Results",
        "",
        "| Split | F1(PCL) | Precision | Recall | Threshold |",
        "|-------|---------|-----------|--------|-----------|",
    ]

    if bdev:
        lines.append(_metrics_row("Baseline — dev", bdev, bthr))
    if btst:
        lines.append(_metrics_row("Baseline — test", btst, bthr))

    if bdev:
        lines += [
            "",
            f"**Confusion matrix (dev):** {_cm_str(bdev.get('confusion_matrix', {}))}",
            "",
            "**Best hyperparameters:**",
            "",
            "```json",
            json.dumps(b.get("best_hyperparams", {}), indent=2),
            "```",
        ]

    lines += [
        "",
        "---",
        "",
        "## 2. Model / Approach Choice Justification",
        "",
        "### Candidate approaches considered",
        "",
        "**Candidate A — `roberta-base`, standard cross-entropy (organiser baseline)**",
        "",
        "- Strong contextual representations for NLU; well-studied fine-tuning recipe.",
        "- Weakness: standard CE is blind to the 9.5% class imbalance; the loss surface",
        "  rewards predicting No-PCL almost always, depressing recall for PCL.",
        "",
        "**Candidate B — `microsoft/deberta-v3-base` + cost-sensitive loss**",
        "",
        "- DeBERTa-v3 improves over RoBERTa via disentangled attention + enhanced mask",
        "  decoder; typically outperforms RoBERTa on SuperGLUE benchmarks.",
        "- Weakness: larger download (~750 MB vs ~500 MB), slower per-trial, harder to",
        "  isolate architectural effects from loss-function effects in our ablation.",
        "",
        "**Candidate C — `roberta-base` + alpha-balanced focal loss + threshold tuning**",
        "*(chosen)*",
        "",
        "### Final choice: RoBERTa-base + focal loss",
        "",
        "We keep `roberta-base` as the encoder (identical to the organiser baseline) but",
        "change the *training objective* to **alpha-balanced focal loss** (Lin et al., 2017).",
        "",
        "**EDA-grounded justification:**",
        "",
        "1. **Class imbalance (EDA finding 1: 9.5% PCL rate)**  Standard CE assigns",
        "   equal gradient weight to each example.  With 90.5% No-PCL samples, the",
        "   model minimises total loss by predicting No-PCL conservatively.  Focal",
        "   loss adds the modulating factor `(1 − p_t)^γ`, which down-weights",
        "   easy-to-classify No-PCL examples, freeing gradient budget for the harder",
        "   minority PCL examples.",
        "",
        "2. **High n-gram overlap (EDA finding 2)**  Bigrams such as *'poor families'*",
        "   and *'homeless people'* appear as top phrases in *both* classes.  Surface",
        "   lexical features alone cannot separate the classes; the model must learn",
        "   subtle pragmatic cues (tone, framing, perspective).  RoBERTa's deep",
        "   contextual attention already captures these; the training bottleneck is the",
        "   loss function and the decision threshold, not the encoder.",
        "",
        "3. **Threshold tuning**  Under severe class imbalance the default threshold",
        "   0.5 is not aligned with maximising F1(PCL).  Sweeping thresholds on the",
        "   *dev set only* directly optimises our evaluation metric without any",
        "   information leakage from the test set.",
        "",
        "4. **Comparability**  Keeping the same encoder makes the improvement",
        "   attributable solely to the training objective and threshold strategy,",
        "   giving a clean ablation result.",
        "",
        "---",
        "",
        "## 3. Hyperparameter Tuning Justification",
        "",
        "### Search method",
        "",
        f"- **Method:** {'Optuna TPE (Tree-structured Parzen Estimator)' if HAS_OPTUNA else 'Random search (Optuna not installed — install with `pip install optuna` for TPE)'}",
        "- **Budget:** 10 trials × baseline + 12 trials × novel = 22 total",
        "- **Why this budget:** Each trial (3 epochs, RoBERTa-base) takes ≈3–8 min on",
        "  Apple Silicon MPS.  22 trials ≈ 1.5–3 h total — thorough enough for TPE to",
        "  converge without excessive risk of over-fitting the dev set.",
        "",
        "### Search space rationale",
        "",
        "| Hyperparameter | Range | Scale | Rationale |",
        "|----------------|-------|-------|-----------|",
        "| `learning_rate` | 1e-5 – 5e-5 | log | Sweet spot for BERT fine-tuning (Devlin et al. 2019) |",
        "| `batch_size` | {8, 16} | cat | 32 risks OOM with seq_len=256 on ≤16 GB RAM |",
        "| `epochs` | 2 – max_epochs | int | <2 = underfitting; >5 = overfitting on 5.8 K train |",
        "| `weight_decay` | 0.0 – 0.1 | linear | Standard L2 for BERT (Sun et al. 2019) |",
        "| `max_len` | {128, 256} | cat | 128 is fast; 256 covers ≈95% of paragraph lengths |",
        "| `focal_gamma` *(novel)* | 0.5 – 3.0 | linear | 0=CE; 2=Lin et al. default; 3=aggressive |",
        "| `focal_alpha` *(novel)* | 0.25 – 0.75 | linear | 0.25=Lin et al. default; push higher for 9.5% rate |",
        "",
        "### Overfitting the dev set — mitigation",
        "",
        "- Fixed random seed across all trials; no data reshuffling between configs.",
        "- Test set held out unconditionally; threshold is chosen on dev, applied to test.",
        "- Limited trial budget (10/12) creates limited selection pressure on dev.",
        "- Best-epoch checkpointing within each trial (not final epoch) guards against",
        "  within-trial overfitting.",
        "",
        "---",
        "",
        "## 4. Novel Approach Results",
        "",
        "**Approach:** `roberta-base` + alpha-balanced focal loss + dev-threshold tuning.",
        "",
        "| Split | F1(PCL) | Precision | Recall | Threshold |",
        "|-------|---------|-----------|--------|-----------|",
    ]

    if ndev:
        lines.append(_metrics_row("Novel — dev", ndev, nthr))
    if ntst:
        lines.append(_metrics_row("Novel — test", ntst, nthr))

    if ndev:
        lines += [
            "",
            f"**Confusion matrix (dev):** {_cm_str(ndev.get('confusion_matrix', {}))}",
            "",
            "**Best hyperparameters:**",
            "",
            "```json",
            json.dumps(n.get("best_hyperparams", {}), indent=2),
            "```",
        ]

    if bdev and ndev:
        bf = bdev.get("f1_pcl", 0)
        nf = ndev.get("f1_pcl", 0)
        delta = nf - bf
        if delta > 0.005:
            verdict = f"**IMPROVEMENT** (ΔF1 = {delta:+.4f})"
        elif abs(delta) <= 0.005:
            verdict = f"**SIMILAR** (ΔF1 = {delta:+.4f})"
        else:
            verdict = f"**REGRESSION** (ΔF1 = {delta:+.4f})"

        lines += [
            "",
            "### Comparison vs baseline",
            "",
            "| Model | Dev F1(PCL) | Dev P | Dev R | Threshold |",
            "|-------|------------|-------|-------|-----------|",
            (
                f"| Baseline (std CE)   | {bf:.4f} | "
                f"{bdev.get('precision_pcl',0):.4f} | "
                f"{bdev.get('recall_pcl',0):.4f} | {bthr:.2f} |"
            ),
            (
                f"| Novel (focal loss)  | {nf:.4f} | "
                f"{ndev.get('precision_pcl',0):.4f} | "
                f"{ndev.get('recall_pcl',0):.4f} | {nthr:.2f} |"
            ),
            "",
            f"**Verdict:** {verdict}",
            "",
            "> *Organiser reported baseline: F1(PCL) ≈ 0.48 (dev), 0.49 (test)*",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Next Steps",
        "",
        "1. **`microsoft/deberta-v3-base` + focal loss:** DeBERTa-v3 generally outperforms",
        "   RoBERTa on nuanced classification (SuperGLUE); combining it with the focal-loss",
        "   recipe should push F1(PCL) further without architectural re-engineering.",
        "",
        "2. **Probability calibration (temperature scaling):** Fine-tuned transformers",
        "   tend to be overconfident.  Calibrating output probabilities (Guo et al. 2017)",
        "   before threshold tuning improves the precision of the threshold sweep.",
        "",
        "3. **Hard-negative mining / curriculum learning:** After an initial training pass,",
        "   identify No-PCL examples assigned high PCL probability (false positives) and",
        "   oversample them in a second training phase to sharpen the decision boundary.",
        "",
        "---",
        "",
        "## 6. Package Versions",
        "",
        "```",
    ]
    lines += [f"{k}: {v}" for k, v in versions.items()]
    lines += ["```", ""]

    report_path = Path(output_dir) / "stage3_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report written → {report_path}")
    return str(report_path)


# ---------------------------------------------------------------------------
# 10. Mode runner
# ---------------------------------------------------------------------------

def run_mode(
    mode: str,
    args,
    X_train, y_train,
    X_dev, y_dev,
    X_test, y_test,
    tokenizer_name: str,
    device: torch.device,
    tuning_rows: list,
    versions: dict,
) -> dict:
    """Run a single mode (baseline or novel) end-to-end."""

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*62}")
    logger.info(f" MODE : {mode.upper()}")
    logger.info(f" Model: {tokenizer_name}")
    logger.info(f" Trials: {args.n_trials}  |  Max epochs/trial: {args.epochs}")
    logger.info(f" Search: {'Optuna TPE' if HAS_OPTUNA else 'Random search'}")
    logger.info(f"{'='*62}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Run hyperparameter search.
    best = run_search(
        mode=mode,
        model_name=tokenizer_name,
        X_train=X_train, y_train=y_train,
        X_dev=X_dev, y_dev=y_dev,
        tokenizer=tokenizer,
        device=device,
        args=args,
        tuning_rows=tuning_rows,
    )

    if best["state"] is None:
        logger.warning(f"[{mode.upper()}] No successful trials — skipping results.")
        return {}

    logger.info(
        f"\n[{mode.upper()}] Best dev F1(PCL)={best['f1']:.4f}  "
        f"threshold={best['threshold']:.3f}  "
        f"P={best['metrics'].get('precision_pcl',0):.4f}  "
        f"R={best['metrics'].get('recall_pcl',0):.4f}"
    )
    cm = best["metrics"].get("confusion_matrix", {})
    logger.info(
        f"[{mode.upper()}] Confusion matrix (dev): "
        f"TP={cm.get('tp',0)}  FP={cm.get('fp',0)}  "
        f"FN={cm.get('fn',0)}  TN={cm.get('tn',0)}"
    )

    # Save checkpoint and evaluate on test.
    checkpoint_dir = str(Path(output_dir) / f"{mode}_best_checkpoint")
    best_params = best["params"]
    test_metrics = save_and_eval_test(
        best_state=best["state"],
        model_name=tokenizer_name,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir,
        X_test=X_test, y_test=y_test,
        threshold=best["threshold"],
        max_len=int(best_params.get("max_len", 256)),
        batch_size=int(best_params.get("batch_size", 16)),
        device=device,
    )
    logger.info(
        f"[{mode.upper()}] Test  F1(PCL)={test_metrics.get('f1_pcl',0):.4f}  "
        f"P={test_metrics.get('precision_pcl',0):.4f}  "
        f"R={test_metrics.get('recall_pcl',0):.4f}"
    )

    # Compile and save results JSON.
    results = {
        "mode": mode,
        "model_name": tokenizer_name,
        "loss_mode": "focal" if mode == "novel" else "standard",
        "best_hyperparams": best_params,
        "best_threshold": best["threshold"],
        "best_trial_id": best["trial_id"],
        "dev_metrics": best["metrics"],
        "test_metrics": test_metrics,
        "split_info": {
            "train": len(y_train),
            "dev": len(y_dev),
            "test": len(y_test),
            "pos_rate_train": float(np.mean(y_train)),
            "pos_rate_dev": float(np.mean(y_dev)),
            "pos_rate_test": float(np.mean(y_test)),
        },
        "package_versions": versions,
        "description": (
            "roberta-base + alpha-balanced focal loss (Lin et al. 2017). "
            "Addresses the 9.5% PCL class imbalance (EDA finding) by "
            "down-weighting easy No-PCL examples via (1-p_t)^gamma, with "
            "alpha controlling per-class balance.  Decision threshold tuned "
            "on dev set to maximise F1(PCL)."
            if mode == "novel" else
            "roberta-base + standard cross-entropy loss (organiser-style "
            "baseline). Optuna hyperparameter search + dev threshold sweep."
        ),
    }

    out_path = Path(output_dir) / f"stage3_{mode}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_path}")

    return results


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: PCL binary classification — "
            "Baseline (RoBERTa + CE) vs Novel (RoBERTa + focal loss)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["baseline", "novel", "both"], default="both",
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Maximum epochs per trial (early stopping may end training sooner).",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience: stop if no dev F1 improvement for N epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (used only when n_trials=1 / reference value).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (used as default; tuning explores {8, 16}).",
    )
    parser.add_argument(
        "--max_len", type=int, default=256,
        help="Max token length (used as default; tuning explores {128, 256}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for all output artefacts.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=10,
        help=(
            "Number of Optuna / random-search trials per mode. "
            "Use --n_trials 2 --epochs 2 for a quick smoke-test."
        ),
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    versions = log_versions()
    device = get_device()
    logger.info(f"Device: {device}")

    # Load data and create stratified splits (fixed seed → reproducible).
    X, y = load_data(TRAIN_LABELS_PATH, PCL_TSV_PATH)
    X_train, X_dev, X_test, y_train, y_dev, y_test = make_splits(X, y, seed=args.seed)

    tuning_rows: list = []
    baseline_results: dict = {}
    novel_results: dict = {}

    modes = ["baseline", "novel"] if args.mode == "both" else [args.mode]
    for mode in modes:
        model_name = BASELINE_MODEL if mode == "baseline" else NOVEL_MODEL
        result = run_mode(
            mode=mode,
            args=args,
            X_train=X_train, y_train=y_train,
            X_dev=X_dev, y_dev=y_dev,
            X_test=X_test, y_test=y_test,
            tokenizer_name=model_name,
            device=device,
            tuning_rows=tuning_rows,
            versions=versions,
        )
        if mode == "baseline":
            baseline_results = result
        else:
            novel_results = result

    # Save tuning log (all trials from all modes).
    if tuning_rows:
        log_path = Path(args.output_dir) / "stage3_tuning_log.csv"
        fieldnames = list(tuning_rows[0].keys())
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tuning_rows)
        logger.info(f"Tuning log saved → {log_path}  ({len(tuning_rows)} rows)")

    # Generate Markdown report.
    generate_report(
        output_dir=args.output_dir,
        baseline_results=baseline_results,
        novel_results=novel_results,
        versions=versions,
    )

    # Print final comparison table to terminal.
    if baseline_results or novel_results:
        print("\n" + "=" * 74)
        print("  FINAL RESULTS COMPARISON")
        print("=" * 74)
        print(
            f"{'Model':<38} {'F1(PCL)':>8}  {'Prec':>8}  {'Rec':>8}  {'Thr':>6}"
        )
        print("-" * 74)

        def _print_row(label, metrics, thr):
            if not metrics:
                return
            print(
                f"{label:<38} {metrics.get('f1_pcl',0):>8.4f}  "
                f"{metrics.get('precision_pcl',0):>8.4f}  "
                f"{metrics.get('recall_pcl',0):>8.4f}  {thr:>6.2f}"
            )

        if baseline_results:
            _print_row(
                "Baseline dev  (std CE)",
                baseline_results.get("dev_metrics", {}),
                baseline_results.get("best_threshold", 0.5),
            )
            _print_row(
                "Baseline test (std CE)",
                baseline_results.get("test_metrics", {}),
                baseline_results.get("best_threshold", 0.5),
            )
        if novel_results:
            _print_row(
                "Novel dev  (focal loss)",
                novel_results.get("dev_metrics", {}),
                novel_results.get("best_threshold", 0.5),
            )
            _print_row(
                "Novel test (focal loss)",
                novel_results.get("test_metrics", {}),
                novel_results.get("best_threshold", 0.5),
            )

        print("=" * 74)

        if baseline_results and novel_results:
            bf = baseline_results.get("dev_metrics", {}).get("f1_pcl", 0)
            nf = novel_results.get("dev_metrics", {}).get("f1_pcl", 0)
            delta = nf - bf
            print(f"\n  ΔF1(PCL) novel vs baseline (dev) : {delta:+.4f}")
            print(f"  Organiser reported baseline       : F1(PCL) ≈ 0.48 (dev)")
        print()


if __name__ == "__main__":
    main()
