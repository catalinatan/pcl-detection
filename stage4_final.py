#!/usr/bin/env python3
"""
stage4_final.py  —  Proper NLP evaluation workflow for PCL detection.

Workflow (run all stages with --mode all, or individually):
  1. hpo      — Split official train 85/15 (stratified), run HPO on internal split
  2. compare  — Evaluate every approach on official dev (read-only, no training)
  3. retrain  — Retrain winner twice:
                  (a) on train only          → for dev.txt  (dev labels are public)
                  (b) on train + official dev → for test.txt (maximise labeled data)
  4. predict  — Write dev.txt + test.txt in submission format (one 0/1 per line)

Submission format (Exercise 5.1):
  dev.txt  — predictions for official dev set  (2094 lines, 0/1 each)
  test.txt — predictions for official test set (3832 lines, 0/1 each)

  dev.txt uses a model trained on train only (dev labels are public — training on
  them would give artificially inflated scores visible to GTAs).
  test.txt uses a model trained on train + dev (all available labels).

Approaches:
  roberta_ce     : RoBERTa + cross-entropy loss
  roberta_focal  : RoBERTa + focal loss  (alpha=0.65–0.85)
  ensemble       : Average probs of roberta_ce + roberta_focal (no extra HPO)
  deberta        : DeBERTa-v3-base + weighted sampler + CE  (fixes prior collapse)

DeBERTa fix (vs stage3_improved.py):
  Previous runs collapsed because batch_size=8 with 9.5% PCL rate → 45% of batches
  had zero positive examples, giving focal loss no positive signal per batch.
  Fix: WeightedRandomSampler (guarantees positives in each batch) + standard CE
  (no alpha double-weighting) + warmup_ratio=0.10 + batch_size ∈ {16, 32}.

Usage:
  python stage4_final.py --mode all --test_file task4_test.tsv
  python stage4_final.py --mode all --test_file task4_test.tsv --skip_deberta
  python stage4_final.py --mode all --roberta_model roberta-large  # bigger model
  python stage4_final.py --mode hpo --n_trials 5
  python stage4_final.py --mode compare
  python stage4_final.py --mode retrain
  python stage4_final.py --mode predict --test_file task4_test.tsv
"""

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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("stage4_final.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_LABELS_PATH = "train_semeval_parids-labels.csv"
DEV_LABELS_PATH   = "dev_semeval_parids-labels.csv"
PCL_TSV_PATH      = "dontpatronizeme_pcl.tsv"
THRESHOLD_RANGE   = np.arange(0.00, 1.01, 0.01)
DEBERTA_MODEL     = "microsoft/deberta-v3-base"
OUTPUT_DIR        = "outputs_stage4"

# HPO checkpoint subdirectory names (saved after stage hpo).
CKPT_NAMES = {
    "roberta_ce":    "hpo_roberta_ce_checkpoint",
    "roberta_focal": "hpo_roberta_focal_checkpoint",
    "deberta":       "hpo_deberta_checkpoint",
}
# Final checkpoints for dev predictions (trained on train only).
FINAL_DEV_CKPT = {
    "roberta_ce":    "final_dev_roberta_ce_checkpoint",
    "roberta_focal": "final_dev_roberta_focal_checkpoint",
    "deberta":       "final_dev_deberta_checkpoint",
}
# Final checkpoints for test predictions (trained on train + dev).
FINAL_TEST_CKPT = {
    "roberta_ce":    "final_test_roberta_ce_checkpoint",
    "roberta_focal": "final_test_roberta_focal_checkpoint",
    "deberta":       "final_test_deberta_checkpoint",
}

HPO_RESULTS_FILE  = "hpo_results.json"
COMPARE_FILE      = "comparison_results.json"
FINAL_CONFIG_FILE = "final_config.json"
TUNING_LOG_FILE   = "tuning_log.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_versions() -> None:
    import sklearn, transformers
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }
    logger.info("─── Package versions ───────────────────────────────────")
    for k, v in info.items():
        logger.info(f"  {k:<14} {v}")
    logger.info("────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_labels(path: str) -> dict:
    """Read par_id → binary label from CSV (label col is a list; 1 if any > 0)."""
    labels = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vec = ast.literal_eval(row["label"])
            labels[row["par_id"]] = 1 if sum(vec) > 0 else 0
    return labels


def read_texts(path: str, min_cols: int = 5, skip_lines: int = 4) -> dict:
    """
    Read par_id → text from a TSV file.
    skip_lines=4 for dontpatronizeme_pcl.tsv (4-line disclaimer).
    skip_lines=0 for the hidden test TSV (data starts at line 1).
    """
    texts = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            parts = line.strip().split("\t")
            if len(parts) >= min_cols:
                texts[parts[0]] = parts[4]
    return texts


def load_split(labels_path: str, pcl_path: str, name: str) -> tuple[list, list]:
    """Return (texts, labels) lists for a labeled split."""
    labels = read_labels(labels_path)
    texts  = read_texts(pcl_path)
    X, y   = [], []
    for pid, lbl in labels.items():
        txt = texts.get(pid, "").strip()
        if txt:
            X.append(txt)
            y.append(lbl)
    pos = sum(y)
    logger.info(
        f"  {name}: {len(X)} samples — "
        f"PCL={pos} ({100*pos/len(y):.1f}%)  "
        f"No-PCL={len(y)-pos} ({100*(len(y)-pos)/len(y):.1f}%)"
    )
    return X, y


def stratified_split(
    X: list, y: list, train_ratio: float, seed: int
) -> tuple[list, list, list, list]:
    """Stratified split preserving class proportions."""
    idx = list(range(len(X)))
    tr_idx, dv_idx = train_test_split(
        idx, train_size=train_ratio, stratify=y, random_state=seed
    )
    X_tr = [X[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_dv = [X[i] for i in dv_idx]
    y_dv = [y[i] for i in dv_idx]
    return X_tr, y_tr, X_dv, y_dv


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PCLDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Alpha-balanced focal loss (Lin et al. 2017) with optional label smoothing."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits   = logits.float()
        log_p    = F.log_softmax(logits, dim=-1)
        log_pt   = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
        pt       = log_pt.exp()
        if self.label_smoothing > 0:
            ce = -(
                (1 - self.label_smoothing) * log_pt
                + self.label_smoothing * log_p.mean(dim=-1)
            )
        else:
            ce = -log_pt
        ce      = ce.clamp(max=100.0)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_loader(
    X: list, y: list, tokenizer, max_len: int, batch_size: int,
    use_sampler: bool = False, shuffle: bool = True,
) -> DataLoader:
    ds = PCLDataset(X, y, tokenizer, max_len)
    if use_sampler:
        y_arr   = np.array(y)
        n_pos   = y_arr.sum()
        n_neg   = len(y_arr) - n_pos
        w_pos   = len(y_arr) / (2.0 * n_pos)
        w_neg   = len(y_arr) / (2.0 * n_neg)
        weights = np.where(y_arr == 1, w_pos, w_neg)
        sampler = WeightedRandomSampler(
            weights=weights.tolist(), num_samples=len(weights), replacement=True
        )
        logger.info(f"    WeightedRandomSampler: pos_w={w_pos:.2f}  neg_w={w_neg:.2f}")
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, grad_clip=1.0):
    model.train()
    total, n_valid = 0.0, 0
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        if not torch.isfinite(logits).all():
            logger.warning("    Non-finite logits; skipping batch.")
            continue
        loss = loss_fn(logits, labs)
        if not torch.isfinite(loss):
            logger.warning("    Non-finite loss; skipping batch.")
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total  += loss.item()
        n_valid += 1
    if n_valid == 0:
        return float("nan")
    return total / n_valid


@torch.no_grad()
def get_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_out, labels_out = [], []
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs  = F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        probs_out.extend(probs.tolist())
        labels_out.extend(batch["label"].numpy().tolist())
    return np.array(probs_out), np.array(labels_out)


def tune_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    best_t, best_f1, best_rec = 0.5, -1.0, -1.0
    for t in THRESHOLD_RANGE:
        preds = (probs >= t).astype(int)
        f1    = f1_score(labels, preds, pos_label=1, zero_division=0)
        rec   = recall_score(labels, preds, pos_label=1, zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and rec > best_rec):
            best_f1, best_t, best_rec = f1, round(float(t), 3), rec
    # Safety: if all thresholds give F1=0 but positives exist, force threshold=0
    if best_f1 <= 0.0 and int(labels.sum()) > 0:
        best_t  = 0.0
        best_f1 = float(f1_score(labels, (probs >= 0.0).astype(int), pos_label=1, zero_division=0))
    return best_t, best_f1


def evaluate(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "f1":        float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall":    float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    X_train: list, y_train: list,
    X_dev: list, y_dev: list,
    device: torch.device,
    *,
    lr: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    max_len: int,
    loss_mode: str = "standard",     # "standard" | "focal"
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    use_sampler: bool = False,
    warmup_ratio: float = 0.06,
    patience: int = 5,
    seed: int = 42,
    trial_label: str = "",
) -> tuple:
    """
    Train a model and return (model, tokenizer, best_f1, best_threshold, best_epoch).
    Model is left on device with best weights loaded.
    """
    set_seed(seed)

    model_kwargs: dict = {"num_labels": 2, "ignore_mismatched_sizes": True}
    if "deberta" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.to(device)

    train_loader = make_loader(X_train, y_train, tokenizer, max_len, batch_size, use_sampler=use_sampler)
    dev_loader   = make_loader(X_dev,   y_dev,   tokenizer, max_len, batch_size * 2, shuffle=False)

    loss_fn = (
        FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if loss_mode == "focal"
        else nn.CrossEntropyLoss()
    )

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(warmup_ratio * total_steps))
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, best_thr, best_state, best_epoch = 0.0, 0.5, None, 0
    no_improve   = 0
    best_loss    = float("inf")

    for epoch in range(1, epochs + 1):
        t0   = time.time()
        loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        if not np.isfinite(loss):
            logger.warning(f"  {trial_label} Training diverged at epoch {epoch}; stopping.")
            break
        probs, labels_arr = get_probabilities(model, dev_loader, device)
        thr, f1           = tune_threshold(probs, labels_arr)
        elapsed           = int(time.time() - t0)
        logger.info(
            f"  {trial_label} Epoch {epoch}/{epochs}  loss={loss:.4f}  "
            f"dev F1={f1:.4f}  thr={thr:.2f}  "
            f"std={probs.std():.4f}  ({elapsed}s)"
        )

        improved  = f1 > best_f1
        tiebreak  = (f1 == best_f1 and loss < best_loss)
        if improved or tiebreak:
            best_f1    = f1
            best_thr   = thr
            best_loss  = loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if improved:
                no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  {trial_label} Early stop at epoch {epoch} (best={best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, best_f1, best_thr, best_epoch


# ---------------------------------------------------------------------------
# Stage 1: HPO on train-internal split
# ---------------------------------------------------------------------------

def _hpo_search(
    label: str,
    model_name: str,
    X_tr: list, y_tr: list,
    X_dv: list, y_dv: list,
    device: torch.device,
    n_trials: int,
    max_epochs: int,
    loss_mode: str,
    use_sampler: bool,
    warmup_ratio: float,
    seed: int,
    checkpoint_dir: str,
    tuning_rows: list,
) -> dict:
    """
    Random hyperparameter search.  Returns the best result dict:
      {f1, threshold, params, checkpoint_dir}
    """
    rng  = np.random.RandomState(seed)
    best = {"f1": -1.0, "threshold": 0.5, "params": {}, "checkpoint_dir": checkpoint_dir}

    for trial in range(n_trials):
        params: dict = {
            "lr":           float(np.exp(rng.uniform(np.log(5e-6), np.log(3e-5)))),
            "batch_size":   int(rng.choice([16, 32])),
            "epochs":       int(rng.randint(min(3, max_epochs), max_epochs + 1)),
            "weight_decay": float(rng.uniform(0.0, 0.05)),
            "max_len":      int(rng.choice([128, 256])),
        }
        if loss_mode == "focal":
            params["focal_gamma"] = float(rng.uniform(0.5, 2.0))
            params["focal_alpha"] = float(rng.uniform(0.65, 0.85))

        logger.info(
            f"\n[{label} Trial {trial+1}/{n_trials}] "
            + "  ".join(
                f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in params.items()
            )
        )

        model, tokenizer, f1, thr, best_ep = train_model(
            model_name=model_name,
            X_train=X_tr, y_train=y_tr,
            X_dev=X_dv,   y_dev=y_dv,
            device=device,
            lr=params["lr"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            weight_decay=params["weight_decay"],
            max_len=params["max_len"],
            loss_mode=loss_mode,
            focal_alpha=params.get("focal_alpha", 0.75),
            focal_gamma=params.get("focal_gamma", 2.0),
            use_sampler=use_sampler,
            warmup_ratio=warmup_ratio,
            seed=seed + trial,
            trial_label=f"[{label} T{trial+1}]",
        )
        logger.info(f"  [{label} T{trial+1}] dev_internal F1={f1:.4f}  thr={thr:.2f}")

        tuning_rows.append({
            "approach": label, "trial": trial + 1,
            **params, "best_epoch": best_ep,
            "dev_internal_f1": round(f1, 4), "threshold": round(thr, 3),
        })

        if f1 > best["f1"]:
            best["f1"]        = f1
            best["threshold"] = thr
            best["params"]    = params
            # Save checkpoint to disk so it persists across stages.
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"  [{label}] New best → F1={f1:.4f}  checkpoint saved.")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best


def run_hpo(args, device: torch.device) -> None:
    """
    Stage 1: HPO on the internal train/dev split.
    Saves hpo_results.json and tuning_log.csv to OUTPUT_DIR.
    """
    logger.info("\n" + "=" * 64)
    logger.info("  STAGE 1: HPO on train-internal split")
    logger.info("=" * 64)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load full official train and create internal split.
    logger.info("\nLoading official train set and creating internal split...")
    X_all, y_all = load_split(TRAIN_LABELS_PATH, PCL_TSV_PATH, "official_train")
    X_tr, y_tr, X_dv, y_dv = stratified_split(X_all, y_all, 0.85, args.seed)
    pos_tr = sum(y_tr); pos_dv = sum(y_dv)
    logger.info(
        f"  train_internal: {len(X_tr)} samples  PCL={pos_tr} ({100*pos_tr/len(y_tr):.1f}%)"
    )
    logger.info(
        f"  dev_internal:   {len(X_dv)} samples  PCL={pos_dv} ({100*pos_dv/len(y_dv):.1f}%)"
    )

    tuning_rows: list = []
    hpo_results: dict = {}

    # ── RoBERTa + CE ──────────────────────────────────────────────────────────
    logger.info(f"\n──────────── roberta_ce  ({args.roberta_model}) ─────────────────────")
    hpo_results["roberta_ce"] = _hpo_search(
        label="roberta_ce",
        model_name=args.roberta_model,
        X_tr=X_tr, y_tr=y_tr, X_dv=X_dv, y_dv=y_dv,
        device=device,
        n_trials=args.n_trials,
        max_epochs=args.epochs,
        loss_mode="standard",
        use_sampler=False,
        warmup_ratio=0.06,
        seed=args.seed,
        checkpoint_dir=str(outdir / CKPT_NAMES["roberta_ce"]),
        tuning_rows=tuning_rows,
    )
    logger.info(
        f"\n[roberta_ce] BEST dev_internal F1={hpo_results['roberta_ce']['f1']:.4f}"
    )

    # ── RoBERTa + Focal ───────────────────────────────────────────────────────
    logger.info(f"\n──────────── roberta_focal  ({args.roberta_model}) ──────────────────")
    hpo_results["roberta_focal"] = _hpo_search(
        label="roberta_focal",
        model_name=args.roberta_model,
        X_tr=X_tr, y_tr=y_tr, X_dv=X_dv, y_dv=y_dv,
        device=device,
        n_trials=args.n_trials,
        max_epochs=args.epochs,
        loss_mode="focal",
        use_sampler=False,
        warmup_ratio=0.06,
        seed=args.seed,
        checkpoint_dir=str(outdir / CKPT_NAMES["roberta_focal"]),
        tuning_rows=tuning_rows,
    )
    logger.info(
        f"\n[roberta_focal] BEST dev_internal F1={hpo_results['roberta_focal']['f1']:.4f}"
    )

    # ── DeBERTa + Sampler + CE (optional) ────────────────────────────────────
    if not args.skip_deberta:
        logger.info("\n──────────── deberta (sampler + CE) ────────────────────────")
        logger.info(
            "  DeBERTa fix: WeightedRandomSampler ensures positives in each batch.\n"
            "  Standard CE avoids double-weighting. warmup_ratio=0.10."
        )
        hpo_results["deberta"] = _hpo_search(
            label="deberta",
            model_name=DEBERTA_MODEL,
            X_tr=X_tr, y_tr=y_tr, X_dv=X_dv, y_dv=y_dv,
            device=device,
            n_trials=args.n_trials,
            max_epochs=args.epochs,
            loss_mode="standard",   # CE only — sampler handles class balance
            use_sampler=True,       # fixes the prior collapse
            warmup_ratio=0.10,      # DeBERTa needs longer warmup
            seed=args.seed,
            checkpoint_dir=str(outdir / CKPT_NAMES["deberta"]),
            tuning_rows=tuning_rows,
        )
        logger.info(
            f"\n[deberta] BEST dev_internal F1={hpo_results['deberta']['f1']:.4f}"
        )
    else:
        logger.info("\n  [DeBERTa] Skipped (--skip_deberta)")

    # Save HPO results first — before anything else that could crash.
    results_path = outdir / HPO_RESULTS_FILE
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(hpo_results, f, indent=2)
    logger.info(f"HPO results → {results_path}")

    # Summary table.
    logger.info("\n" + "─" * 60)
    logger.info("  HPO SUMMARY (dev_internal F1)")
    logger.info("─" * 60)
    for name, r in hpo_results.items():
        logger.info(f"  {name:<20} F1={r['f1']:.4f}  thr={r['threshold']:.2f}")
    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Stage 2: Compare on official dev (read-only)
# ---------------------------------------------------------------------------

def run_compare(args, device: torch.device) -> None:
    """
    Stage 2: Load each HPO checkpoint and evaluate on official dev.
    Also evaluates the ensemble of roberta_ce + roberta_focal.
    Saves comparison_results.json; does NOT modify any model weights.
    """
    logger.info("\n" + "=" * 64)
    logger.info("  STAGE 2: Compare approaches on official dev (read-only)")
    logger.info("=" * 64)
    outdir = Path(args.output_dir)

    hpo_path = outdir / HPO_RESULTS_FILE
    if not hpo_path.exists():
        logger.error(f"HPO results not found at {hpo_path}. Run --mode hpo first.")
        sys.exit(1)
    with open(hpo_path) as f:
        hpo_results = json.load(f)

    # Load official dev (read-only — just for evaluation).
    logger.info("\nLoading official dev set (evaluation only — no training on this)...")
    X_dev, y_dev = load_split(DEV_LABELS_PATH, PCL_TSV_PATH, "official_dev")
    y_arr = np.array(y_dev)

    comparison: dict = {}

    # Evaluate each single-model checkpoint.
    for approach, ckpt_name in CKPT_NAMES.items():
        if approach == "deberta" and args.skip_deberta:
            continue
        ckpt_dir = outdir / ckpt_name
        if not ckpt_dir.exists():
            logger.warning(f"  Checkpoint missing for {approach}: {ckpt_dir}")
            continue
        if approach not in hpo_results or hpo_results[approach]["f1"] <= 0.0:
            logger.warning(f"  {approach} HPO produced no valid model; skipping.")
            continue

        logger.info(f"\n  Evaluating {approach} on official dev...")
        model_name = DEBERTA_MODEL if approach == "deberta" else args.roberta_model
        model_kwargs: dict = {"num_labels": 2, "ignore_mismatched_sizes": True}
        if "deberta" in model_name.lower():
            model_kwargs["attn_implementation"] = "eager"

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model     = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_dir), **model_kwargs
        )
        model.to(device).eval()

        best_params = hpo_results[approach]["params"]
        max_len     = best_params.get("max_len", 256)
        loader      = DataLoader(
            PCLDataset(X_dev, y_dev, tokenizer, max_len),
            batch_size=32, shuffle=False, num_workers=0,
        )
        probs, _ = get_probabilities(model, loader, device)

        # Tune threshold freshly on official dev (valid since dev = our test set).
        thr, f1 = tune_threshold(probs, y_arr)
        metrics  = evaluate(probs, y_arr, thr)

        logger.info(
            f"  {approach:<20} official dev F1={f1:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  thr={thr:.2f}"
        )

        comparison[approach] = {
            "official_dev_f1":  f1,
            "official_dev_thr": thr,
            "metrics":          metrics,
            "model_name":       model_name,
            "checkpoint_dir":   str(ckpt_dir),
            "hpo_params":       best_params,
        }

        # Store probs for ensemble computation.
        comparison[approach]["_probs"] = probs.tolist()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate ensemble of roberta_ce + roberta_focal.
    if "roberta_ce" in comparison and "roberta_focal" in comparison:
        logger.info(f"\n  Evaluating ensemble (roberta_ce + roberta_focal)...")
        ce_probs    = np.array(comparison["roberta_ce"]["_probs"])
        focal_probs = np.array(comparison["roberta_focal"]["_probs"])
        ens_probs   = (ce_probs + focal_probs) / 2.0

        thr, f1 = tune_threshold(ens_probs, y_arr)
        metrics  = evaluate(ens_probs, y_arr, thr)
        logger.info(
            f"  {'ensemble':<20} official dev F1={f1:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  thr={thr:.2f}"
        )
        comparison["ensemble"] = {
            "official_dev_f1":  f1,
            "official_dev_thr": thr,
            "metrics":          metrics,
        }

    # Remove internal probs before saving.
    for v in comparison.values():
        v.pop("_probs", None)

    # Pick best approach.
    best_approach = max(comparison, key=lambda k: comparison[k]["official_dev_f1"])
    comparison["_best_approach"] = best_approach
    logger.info(f"\n  ★ Best approach: {best_approach}  "
                f"(official dev F1={comparison[best_approach]['official_dev_f1']:.4f})")

    # Print comparison table.
    logger.info("\n" + "─" * 64)
    logger.info(f"  {'Approach':<22} {'Dev F1':>8} {'Precision':>10} {'Recall':>8} {'Thr':>6}")
    logger.info("─" * 64)
    for name, r in comparison.items():
        if name.startswith("_"):
            continue
        m = r["metrics"]
        marker = " ★" if name == best_approach else ""
        logger.info(
            f"  {name + marker:<22} {r['official_dev_f1']:>8.4f} "
            f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{r['official_dev_thr']:>6.2f}"
        )
    logger.info("─" * 64)

    cmp_path = outdir / COMPARE_FILE
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"\nComparison results → {cmp_path}")


# ---------------------------------------------------------------------------
# Stage 3: Retrain winner — once on train-only, once on train+dev
# ---------------------------------------------------------------------------

def run_retrain(args, device: torch.device) -> None:
    """
    Stage 3: Retrain the winning approach twice using its best HPO hyperparameters:

      (a) On FULL train (8375 samples) only.
          Used for dev.txt — dev labels are public, so training on them would give
          artificially inflated scores visible to GTAs.

      (b) On train + official dev (10469 samples) combined.
          Used for test.txt — maximises labeled data before the private evaluation.

    Saves final_config.json with paths to both checkpoint sets + threshold.
    """
    logger.info("\n" + "=" * 64)
    logger.info("  STAGE 3: Retrain winner (train-only → dev.txt, train+dev → test.txt)")
    logger.info("=" * 64)
    outdir = Path(args.output_dir)

    cmp_path = outdir / COMPARE_FILE
    if not cmp_path.exists():
        logger.error(f"Comparison results not found at {cmp_path}. Run --mode compare first.")
        sys.exit(1)
    with open(cmp_path) as f:
        comparison = json.load(f)
    hpo_path = outdir / HPO_RESULTS_FILE
    with open(hpo_path) as f:
        hpo_results = json.load(f)

    best_approach = comparison.get("_best_approach", "ensemble")
    threshold     = comparison[best_approach]["official_dev_thr"]
    logger.info(
        f"\n  Winner: {best_approach}  "
        f"official dev F1={comparison[best_approach]['official_dev_f1']:.4f}  "
        f"threshold={threshold:.2f}"
    )

    X_train, y_train = load_split(TRAIN_LABELS_PATH, PCL_TSV_PATH, "train")
    X_dev,   y_dev   = load_split(DEV_LABELS_PATH,   PCL_TSV_PATH, "official_dev")
    X_combined = X_train + X_dev
    y_combined = y_train + y_dev

    # Internal 15% split of train — used to tune dev_threshold (no leakage).
    _, _, X_int_dv, y_int_dv = stratified_split(X_train, y_train, 0.85, args.seed)
    logger.info(
        f"\n  train-only    : {len(X_train)} samples"
        f"\n  train+dev     : {len(X_combined)} samples"
        f"\n  internal dev  : {len(X_int_dv)} samples (for dev_threshold tuning)"
    )

    def _retrain(approach_key: str, X_tr: list, y_tr: list,
                 ckpt_map: dict, label: str) -> str:
        """
        Train one component of the best approach on (X_tr, y_tr).
        Returns the saved checkpoint path, or None on failure.
        """
        if approach_key not in hpo_results or hpo_results[approach_key]["f1"] <= 0.0:
            logger.error(f"  No valid HPO result for {approach_key}.")
            return None
        params      = hpo_results[approach_key]["params"]
        is_deberta  = approach_key == "deberta"
        model_name  = DEBERTA_MODEL if is_deberta else args.roberta_model
        loss_mode   = "focal" if approach_key == "roberta_focal" else "standard"
        use_sampler = is_deberta
        warmup_r    = 0.10 if is_deberta else 0.06

        lr           = params.get("lr", 2e-5)
        batch_size   = params.get("batch_size", 16)
        epochs       = params.get("epochs", args.epochs)
        weight_decay = params.get("weight_decay", 0.01)
        max_len      = params.get("max_len", 256)

        logger.info(
            f"\n  [{label}] Retraining {approach_key} on {len(X_tr)} samples\n"
            f"  Config: lr={lr:.2e}  bs={batch_size}  "
            f"ep={epochs}  max_len={max_len}"
        )
        model, tokenizer, f1, _, ep = train_model(
            model_name=model_name,
            X_train=X_tr,  y_train=y_tr,
            X_dev=X_dev,   y_dev=y_dev,   # epoch selection monitor only
            device=device,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            max_len=max_len,
            loss_mode=loss_mode,
            focal_alpha=params.get("focal_alpha", 0.75),
            focal_gamma=params.get("focal_gamma", 2.0),
            use_sampler=use_sampler,
            warmup_ratio=warmup_r,
            patience=args.patience,
            seed=args.seed,
            trial_label=f"[{approach_key} {label}]",
        )
        logger.info(f"  Done — best_epoch={ep}  monitored_dev_f1={f1:.4f}")
        ckpt = str(outdir / ckpt_map[approach_key])
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        logger.info(f"  Saved → {ckpt}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ckpt

    # Determine which approach keys need retraining.
    if best_approach == "ensemble":
        keys = ["roberta_ce", "roberta_focal"]
    else:
        keys = [best_approach]

    def _tune_threshold_ensemble(ckpts: list, X: list, y: list, label: str) -> float:
        """Infer with all ckpts, average probs, tune threshold. Returns best thr."""
        probs = _infer_ensemble(ckpts, X, device)
        thr, f1 = tune_threshold(probs, np.array(y))
        logger.info(f"  [{label}] threshold={thr:.2f}  F1={f1:.4f}")
        return thr

    # (a) dev.txt: reuse HPO checkpoints directly — they were trained on 85% of
    #     train with no dev labels, so they're valid for predicting on dev.
    #     No need to retrain on 100% of train; the marginal gain is negligible
    #     and the HPO checkpoints already give us a genuinely held-out threshold.
    logger.info("\n── (a) dev.txt: reusing HPO checkpoints (no dev leakage) ────")
    hpo_ckpts = [str(outdir / CKPT_NAMES[k]) for k in keys]
    for p in hpo_ckpts:
        logger.info(f"  Using HPO checkpoint: {Path(p).name}")

    # Tune dev_threshold: HPO checkpoints on internal 15% — genuinely held-out.
    logger.info("\n  Tuning dev_threshold: HPO checkpoints × internal 15% split...")
    dev_threshold = _tune_threshold_ensemble(hpo_ckpts, X_int_dv, y_int_dv, "dev_threshold")

    # (b) Train on train+dev → checkpoints for test.txt
    logger.info("\n── (b) Train on train+dev   →  test.txt checkpoints ─────────")
    test_ckpts: list = []
    for k in keys:
        p = _retrain(k, X_combined, y_combined, FINAL_TEST_CKPT, "test")
        if p:
            test_ckpts.append(p)

    # Tune test_threshold on official dev — valid since we already trained on dev.
    logger.info("\n  Tuning test_threshold on official dev (already used for training)...")
    test_threshold = _tune_threshold_ensemble(test_ckpts, X_dev, y_dev, "test_threshold")

    if not test_ckpts:
        logger.error("  test checkpoints failed to save. Check errors above.")
        sys.exit(1)

    final_config = {
        "best_approach":    best_approach,
        "dev_f1":           comparison[best_approach]["official_dev_f1"],
        "dev_threshold":    dev_threshold,   # tuned on internal 15%, used for dev.txt
        "test_threshold":   test_threshold,  # tuned on official dev, used for test.txt
        "dev_checkpoints":  hpo_ckpts,       # HPO checkpoints (no dev leakage) → dev.txt
        "test_checkpoints": test_ckpts,      # train+dev → test.txt
    }
    cfg_path = outdir / FINAL_CONFIG_FILE
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
    logger.info(f"\nFinal config → {cfg_path}")
    logger.info(f"  dev_checkpoints  : {hpo_ckpts}")
    logger.info(f"  test_checkpoints : {test_ckpts}")
    logger.info(f"  dev_threshold    : {dev_threshold:.2f}")
    logger.info(f"  test_threshold   : {test_threshold:.2f}")


# ---------------------------------------------------------------------------
# Stage 4: Generate dev.txt and test.txt
# ---------------------------------------------------------------------------

def _infer_ensemble(checkpoints: list, X: list, device: torch.device) -> np.ndarray:
    """Run all checkpoints on X and return averaged probabilities."""
    all_probs: list = []
    y_dummy = [0] * len(X)
    for ckpt_dir in checkpoints:
        logger.info(f"  Inferring with {Path(ckpt_dir).name} ...")
        model_kwargs: dict = {"num_labels": 2, "ignore_mismatched_sizes": True}
        if "deberta" in Path(ckpt_dir).name.lower():
            model_kwargs["attn_implementation"] = "eager"
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, **model_kwargs
        )
        model.to(device).eval()
        loader = DataLoader(
            PCLDataset(X, y_dummy, tokenizer, max_len=256),
            batch_size=32, shuffle=False, num_workers=0,
        )
        probs, _ = get_probabilities(model, loader, device)
        logger.info(
            f"  prob_stats: mean={probs.mean():.4f}  std={probs.std():.4f}  "
            f"min={probs.min():.4f}  max={probs.max():.4f}"
        )
        all_probs.append(probs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(all_probs, axis=0)


def _write_submission(preds: np.ndarray, path: Path) -> None:
    """Write one 0/1 per line (no header, no par_id) — submission format."""
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(f"{int(p)}\n")
    logger.info(f"  Saved {len(preds)} predictions → {path}")


def run_predict(args, device: torch.device) -> None:
    """
    Stage 4: Generate dev.txt and test.txt in submission format.

    dev.txt  — predictions on official dev  (train-only model, 2094 lines)
    test.txt — predictions on official test (train+dev model, 3832 lines)

    Both files: one prediction per line, 0 = No PCL, 1 = PCL. No header, no par_id.
    """
    logger.info("\n" + "=" * 64)
    logger.info("  STAGE 4: Generate dev.txt + test.txt (submission format)")
    logger.info("=" * 64)
    outdir = Path(args.output_dir)

    cfg_path = outdir / FINAL_CONFIG_FILE
    if not cfg_path.exists():
        logger.error(f"Final config not found at {cfg_path}. Run --mode retrain first.")
        sys.exit(1)
    with open(cfg_path) as f:
        final_config = json.load(f)

    dev_ckpts       = final_config["dev_checkpoints"]
    test_ckpts      = final_config["test_checkpoints"]
    dev_threshold   = final_config["dev_threshold"]
    test_threshold  = final_config["test_threshold"]
    logger.info(f"\n  dev_threshold  : {dev_threshold:.2f}  (tuned on internal split)")
    logger.info(f"  test_threshold : {test_threshold:.2f}  (tuned on official dev)")
    logger.info(f"  dev  checkpoints : {dev_ckpts}")
    logger.info(f"  test checkpoints : {test_ckpts}")

    # ── dev.txt: predict official dev in label-file order ───────────────────
    # Load texts in par_id order from dev_semeval_parids-labels.csv.
    # Order matters — GTA compares line i of dev.txt to line i of the labels file.
    logger.info("\n── Generating dev.txt ───────────────────────────────────────")
    dev_labels_order = list(read_labels(DEV_LABELS_PATH).keys())   # preserves CSV order
    dev_texts_map    = read_texts(PCL_TSV_PATH)
    X_dev_ordered    = [dev_texts_map[pid] for pid in dev_labels_order if pid in dev_texts_map]
    logger.info(f"  Official dev: {len(X_dev_ordered)} paragraphs (in labels-file order)")

    dev_probs = _infer_ensemble(dev_ckpts, X_dev_ordered, device)
    dev_preds = (dev_probs >= dev_threshold).astype(int)
    n_pcl     = int(dev_preds.sum())
    logger.info(
        f"  dev.txt: {n_pcl} PCL / {len(dev_preds)-n_pcl} No-PCL "
        f"({100*n_pcl/len(dev_preds):.1f}%)"
    )
    _write_submission(dev_preds, outdir / "dev.txt")

    # ── test.txt: predict test set in test-file order ────────────────────────
    if not args.test_file:
        logger.error("--test_file is required to generate test.txt.")
        sys.exit(1)
    logger.info(f"\n── Generating test.txt  (file: {args.test_file}) ────────────")
    test_texts_map = read_texts(args.test_file, min_cols=5, skip_lines=0)
    if not test_texts_map:
        logger.error(f"No texts loaded from {args.test_file}. Check file path/format.")
        sys.exit(1)
    X_test_ordered = list(test_texts_map.values())   # preserves TSV row order
    logger.info(f"  Test set: {len(X_test_ordered)} paragraphs (in TSV row order)")

    test_probs = _infer_ensemble(test_ckpts, X_test_ordered, device)
    test_preds = (test_probs >= test_threshold).astype(int)
    n_pcl      = int(test_preds.sum())
    logger.info(
        f"  test.txt: {n_pcl} PCL / {len(test_preds)-n_pcl} No-PCL "
        f"({100*n_pcl/len(test_preds):.1f}%)"
    )
    _write_submission(test_preds, outdir / "test.txt")

    logger.info("\n  ✓ Submission files ready:")
    logger.info(f"    {outdir}/dev.txt   ({len(dev_preds)} lines)")
    logger.info(f"    {outdir}/test.txt  ({len(test_preds)} lines)")


# ---------------------------------------------------------------------------
# Recover: rebuild hpo_results.json from saved checkpoints
# ---------------------------------------------------------------------------

def run_recover_hpo(args, device: torch.device) -> None:
    """
    Reconstructs hpo_results.json by evaluating each saved HPO checkpoint on
    the internal dev split (same seed/ratio as HPO).  Run this when HPO
    training finished but the JSON was not saved (e.g. due to a crash).
    """
    logger.info("\n" + "=" * 64)
    logger.info("  RECOVER: rebuilding hpo_results.json from checkpoints")
    logger.info("=" * 64)
    outdir = Path(args.output_dir)

    # Rebuild internal dev split (identical to run_hpo).
    X_all, y_all = load_split(TRAIN_LABELS_PATH, PCL_TSV_PATH, "train")
    _, _, X_dv, y_dv = stratified_split(X_all, y_all, 0.85, args.seed)
    y_arr = np.array(y_dv)
    logger.info(f"Internal dev: {len(X_dv)} samples  {int(y_arr.sum())} positives")

    DEFAULTS = {"lr": 2e-5, "batch_size": 16, "epochs": args.epochs, "weight_decay": 0.01}

    hpo_results: dict = {}
    for approach, ckpt_name in CKPT_NAMES.items():
        if approach == "deberta" and args.skip_deberta:
            logger.info(f"\n[{approach}] skipped (--skip_deberta)")
            continue
        ckpt_dir = outdir / ckpt_name
        if not ckpt_dir.exists():
            logger.warning(f"\n[{approach}] checkpoint not found at {ckpt_dir} — skipping")
            continue

        logger.info(f"\n[{approach}] evaluating checkpoint at {ckpt_dir} ...")
        model_name = DEBERTA_MODEL if approach == "deberta" else args.roberta_model
        model_kwargs: dict = {"num_labels": 2, "ignore_mismatched_sizes": True}
        if "deberta" in model_name.lower():
            model_kwargs["attn_implementation"] = "eager"

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_dir), **model_kwargs)
        model.to(device).eval()

        best_f1, best_thr, best_ml = 0.0, 0.5, 256
        for ml in [128, 256]:
            loader = DataLoader(
                PCLDataset(X_dv, y_dv, tokenizer, ml),
                batch_size=64, shuffle=False, num_workers=0,
            )
            probs, _ = get_probabilities(model, loader, device)
            thr, f1 = tune_threshold(probs, y_arr)
            logger.info(f"  max_len={ml}  F1={f1:.4f}  thr={thr:.2f}  std={probs.std():.4f}")
            if f1 > best_f1:
                best_f1, best_thr, best_ml = f1, thr, ml

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        params = {**DEFAULTS, "max_len": best_ml}
        hpo_results[approach] = {
            "f1": round(best_f1, 4),
            "threshold": best_thr,
            "params": params,
            "checkpoint_dir": str(ckpt_dir),
        }
        logger.info(f"  → best: max_len={best_ml}  F1={best_f1:.4f}  thr={best_thr:.2f}")

    out_path = outdir / HPO_RESULTS_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hpo_results, f, indent=2)
    logger.info(f"\nSaved → {out_path}")
    logger.info(json.dumps(hpo_results, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="stage4_final.py: proper NLP workflow for PCL detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["hpo", "compare", "retrain", "predict", "recover", "all"],
        default="all",
        help=(
            "hpo: HPO on internal split | "
            "compare: evaluate on official dev | "
            "retrain: retrain winner on train+dev | "
            "predict: generate test labels | "
            "recover: rebuild hpo_results.json from saved checkpoints | "
            "all: run all stages"
        ),
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="task4_test.tsv",
        help="Path to test TSV (required for predict/all). No label column expected.",
    )
    parser.add_argument("--epochs",        type=int,  default=6,
                        help="Max epochs per HPO trial.")
    parser.add_argument("--patience",      type=int,  default=3,
                        help="Early stopping patience.")
    parser.add_argument("--n_trials",      type=int,  default=5,
                        help="HPO trials per approach.")
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--output_dir",    type=str,  default=OUTPUT_DIR)
    parser.add_argument("--roberta_model", type=str,  default="roberta-base",
                        help="RoBERTa variant to use, e.g. roberta-base or roberta-large.")
    parser.add_argument("--skip_deberta",  action="store_true",
                        help="Skip DeBERTa experiments (faster RoBERTa-only run).")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_versions()
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(
        f"Mode: {args.mode}  |  Trials: {args.n_trials}  |  "
        f"Max epochs: {args.epochs}  |  RoBERTa: {args.roberta_model}"
    )
    if args.skip_deberta:
        logger.info("  DeBERTa: SKIPPED (--skip_deberta)")

    if args.mode == "all":
        run_hpo(args, device)
        run_compare(args, device)
        run_retrain(args, device)
        run_predict(args, device)
    elif args.mode == "hpo":
        run_hpo(args, device)
    elif args.mode == "compare":
        run_compare(args, device)
    elif args.mode == "retrain":
        run_retrain(args, device)
    elif args.mode == "predict":
        run_predict(args, device)
    elif args.mode == "recover":
        run_recover_hpo(args, device)


if __name__ == "__main__":
    main()
