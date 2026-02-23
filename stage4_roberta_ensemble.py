#!/usr/bin/env python3
"""
stage4_roberta_ensemble.py — Multi-RoBERTa ensemble (base + large).

Motivation: DeBERTa experiments collapsed. RoBERTa-base ensemble in
stage3_improved.py already reached dev F1=0.5966. This file pushes further
by adding roberta-large and ensembling 5 diverse RoBERTa configurations.

5 fixed model configs (no random search — deterministic and resume-friendly):
    1. rl_focal      roberta-large  focal α=0.75  seed=42   lr=1e-5  bs=8
    2. rl_ce         roberta-large  standard CE   seed=123  lr=8e-6  bs=8
    3. rb_focal      roberta-base   focal α=0.75  seed=42   lr=2e-5  bs=16
    4. rb_focal_b    roberta-base   focal α=0.65  seed=456  lr=3e-5  bs=16
    5. rb_ce_smooth  roberta-base   CE+smooth     seed=789  lr=2e-5  bs=16

No weighted sampler on any model — focal loss alone handles the 9.5% class
imbalance. (Combining sampler + focal causes gradient collapse; see comments
in stage3_improved.py.)

Usage:
    python stage4_roberta_ensemble.py --mode all           # train all 5, then ensemble
    python stage4_roberta_ensemble.py --mode large         # train rl_focal + rl_ce only
    python stage4_roberta_ensemble.py --mode base          # train rb_* only
    python stage4_roberta_ensemble.py --mode ensemble      # ensemble saved checkpoints
    python stage4_roberta_ensemble.py --mode predict --test_file /path/to/task4_test.tsv

Outputs → outputs_stage4/
  Each model: outputs_stage4/{name}_checkpoint/
  Predictions: outputs_stage4/test_predictions.csv
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Logging & constants
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAIN_LABELS_PATH = "train_semeval_parids-labels.csv"
DEV_LABELS_PATH   = "dev_semeval_parids-labels.csv"
PCL_TSV_PATH      = "dontpatronizeme_pcl.tsv"

THRESHOLD_RANGE = np.arange(0.00, 1.01, 0.01)

# Prior results for the comparison table (from stage3.py and stage3_improved.py).
PRIOR_RESULTS = {
    "Baseline (RoBERTa-base + CE)": {
        "dev_f1": 0.5865, "dev_p": 0.6854, "dev_r": 0.5126,
    },
    "Novel (RoBERTa-base + focal)": {
        "dev_f1": 0.5887, "dev_p": 0.5659, "dev_r": 0.6134,
    },
    "Ensemble stage3_improved (base+novel)": {
        "dev_f1": 0.5966, "dev_p": 0.5206, "dev_r": 0.6985,
    },
}

# ---------------------------------------------------------------------------
# 2. Five fixed diverse model configurations
# ---------------------------------------------------------------------------
# Note on rb_ce_smooth: uses loss_mode="focal" with gamma=0 and alpha=0.5,
# which degenerates to 0.5 * label-smoothed CE. Equivalent to CE+smooth
# for optimisation purposes and avoids adding a separate code path.
CONFIGS = [
    {
        "name":            "rl_focal",
        "model_name":      "roberta-large",
        "loss_mode":       "focal",
        "focal_alpha":     0.75,
        "focal_gamma":     2.0,
        "label_smoothing": 0.0,
        "seed":            42,
        "lr":              1e-5,
        "batch_size":      8,
        "max_len":         256,
        "weight_decay":    0.01,
        "epochs":          5,
    },
    {
        "name":            "rl_ce",
        "model_name":      "roberta-large",
        "loss_mode":       "standard",
        "focal_alpha":     0.5,    # unused for standard CE
        "focal_gamma":     2.0,
        "label_smoothing": 0.0,
        "seed":            123,
        "lr":              8e-6,
        "batch_size":      8,
        "max_len":         256,
        "weight_decay":    0.01,
        "epochs":          5,
    },
    {
        "name":            "rb_focal",
        "model_name":      "roberta-base",
        "loss_mode":       "focal",
        "focal_alpha":     0.75,
        "focal_gamma":     2.0,
        "label_smoothing": 0.0,
        "seed":            42,
        "lr":              2e-5,
        "batch_size":      16,
        "max_len":         256,
        "weight_decay":    0.01,
        "epochs":          4,
    },
    {
        "name":            "rb_focal_b",
        "model_name":      "roberta-base",
        "loss_mode":       "focal",
        "focal_alpha":     0.65,
        "focal_gamma":     2.0,
        "label_smoothing": 0.0,
        "seed":            456,
        "lr":              3e-5,
        "batch_size":      16,
        "max_len":         256,
        "weight_decay":    0.01,
        "epochs":          4,
    },
    {
        "name":            "rb_ce_smooth",
        "model_name":      "roberta-base",
        "loss_mode":       "focal",   # gamma=0, alpha=0.5 → equivalent to CE + smooth
        "focal_alpha":     0.5,
        "focal_gamma":     0.0,
        "label_smoothing": 0.1,
        "seed":            789,
        "lr":              2e-5,
        "batch_size":      16,
        "max_len":         256,
        "weight_decay":    0.01,
        "epochs":          4,
    },
]

# Which configs run in each mode.
MODE_CONFIGS = {
    "large":   ["rl_focal", "rl_ce"],
    "base":    ["rb_focal", "rb_focal_b", "rb_ce_smooth"],
    "all":     ["rl_focal", "rl_ce", "rb_focal", "rb_focal_b", "rb_ce_smooth"],
}

# Checkpoint suffix appended to each config name.
CKPT_SUFFIX = "_checkpoint"


# ---------------------------------------------------------------------------
# 3. Utilities
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
    versions = {
        "python":       sys.version.split()[0],
        "torch":        torch.__version__,
        "transformers": transformers.__version__,
        "sklearn":      sklearn.__version__,
        "numpy":        np.__version__,
    }
    logger.info("─── Package versions ───────────────────────────")
    for k, v in versions.items():
        logger.info(f"  {k:<14} {v}")
    logger.info("────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------

def read_labels(path: str) -> dict:
    labels = {}
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec = ast.literal_eval(row["label"])
            labels[row["par_id"]] = 1 if sum(vec) > 0 else 0
    return labels


def read_texts(path: str, min_cols: int = 5, skip_lines: int = 4) -> dict:
    """
    skip_lines=4  for dontpatronizeme_pcl.tsv (4-line disclaimer header).
    skip_lines=0  for the hidden test TSV (data starts at line 1, no header).
    min_cols=5    accepts files without a label column.
    """
    texts = {}
    with open(path, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            parts = line.strip().split("\t")
            if len(parts) >= min_cols:
                texts[parts[0]] = parts[4]
    return texts


def load_data(labels_path: str, pcl_path: str, split_name: str):
    labels = read_labels(labels_path)
    texts  = read_texts(pcl_path)
    X, y = [], []
    for par_id, label in labels.items():
        text = texts.get(par_id, "").strip()
        if text:
            X.append(text)
            y.append(label)
    n_pcl = sum(y)
    logger.info(
        f"Loaded {split_name}={len(X)} samples — "
        f"PCL={n_pcl} ({100*n_pcl/len(y):.1f}%)  "
        f"No-PCL={len(y)-n_pcl} ({100*(len(y)-n_pcl)/len(y):.1f}%)"
    )
    return X, y


# ---------------------------------------------------------------------------
# 5. Dataset
# ---------------------------------------------------------------------------

class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
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
# 6. Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Alpha-balanced focal loss (Lin et al. 2017) with optional label smoothing."""

    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        logits      = logits.float()
        log_softmax = F.log_softmax(logits, dim=-1)
        log_pt      = log_softmax.gather(1, targets.view(-1, 1)).squeeze(1)
        pt          = log_pt.exp()

        if self.label_smoothing > 0:
            ce_loss = -(
                (1.0 - self.label_smoothing) * log_pt
                + self.label_smoothing * log_softmax.mean(dim=-1)
            )
        else:
            ce_loss = -log_pt

        ce_loss = ce_loss.clamp(max=100.0)

        alpha_t = torch.where(
            targets == 1,
            torch.full_like(pt, self.alpha),
            torch.full_like(pt, 1.0 - self.alpha),
        )
        focal = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ---------------------------------------------------------------------------
# 7. Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device,
                grad_clip=1.0, scaler=None, use_amp=False):
    model.train()
    _use_amp   = (use_amp or scaler is not None) and device.type == "cuda"
    _amp_dtype = torch.float16 if scaler is not None else torch.bfloat16
    total_loss, n_valid = 0.0, 0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=_use_amp, dtype=_amp_dtype):
            logits = model(input_ids=ids, attention_mask=mask).logits
            if not torch.isfinite(logits).all():
                logger.warning("    Non-finite logits; skipping batch.")
                continue
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                logger.warning("    Non-finite loss; skipping batch.")
                continue
        if _use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_valid    += 1
    if n_valid == 0:
        logger.warning("    All batches had non-finite loss; epoch failed.")
        return float("nan")
    return total_loss / n_valid


@torch.no_grad()
def get_probabilities(model, loader, device, use_amp=False):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast(
            device_type="cuda", enabled=use_amp and device.type == "cuda"
        ):
            logits = model(input_ids=ids, attention_mask=mask).logits
        probs = F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def tune_threshold(probs, labels):
    labels = np.asarray(labels)
    best_t, best_f1 = 0.5, -1.0
    best_recall, best_precision = -1.0, -1.0
    for t in THRESHOLD_RANGE:
        preds     = (probs >= t).astype(int)
        f1        = f1_score(labels, preds, pos_label=1, zero_division=0)
        recall    = recall_score(labels, preds, pos_label=1, zero_division=0)
        precision = precision_score(labels, preds, pos_label=1, zero_division=0)
        if (
            f1 > best_f1
            or (f1 == best_f1 and recall > best_recall)
            or (f1 == best_f1 and recall == best_recall and precision > best_precision)
        ):
            best_f1        = f1
            best_t         = round(float(t), 3)
            best_recall    = recall
            best_precision = precision
    if best_f1 <= 0.0 and int(labels.sum()) > 0:
        best_t    = 0.0
        preds     = (probs >= best_t).astype(int)
        best_f1   = f1_score(labels, preds, pos_label=1, zero_division=0)
    return best_t, best_f1


def evaluate_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return {
        "f1_pcl":        float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "precision_pcl": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall_pcl":    float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------------
# 8. Core training loop (identical to stage3_improved.py with all fixes)
# ---------------------------------------------------------------------------

def run_training_improved(
    model_name,
    X_train, y_train,
    X_dev, y_dev,
    tokenizer,
    *,
    lr, batch_size, epochs, weight_decay, max_len,
    seed, device,
    loss_mode="focal",
    focal_gamma=2.0, focal_alpha=0.25,
    label_smoothing=0.0,
    use_weighted_sampler=False,
    freeze_layers=0,
    patience=3,
):
    set_seed(seed)

    train_ds = PCLDataset(X_train, y_train, tokenizer, max_len)
    dev_ds   = PCLDataset(X_dev,   y_dev,   tokenizer, max_len)
    pin      = device.type == "cuda"

    if use_weighted_sampler:
        y_arr       = np.array(y_train)
        n_pos       = y_arr.sum()
        n_neg       = len(y_arr) - n_pos
        weight_pos  = len(y_arr) / (2.0 * n_pos)
        weight_neg  = len(y_arr) / (2.0 * n_neg)
        sample_w    = np.where(y_arr == 1, weight_pos, weight_neg)
        sampler     = WeightedRandomSampler(sample_w.tolist(), len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=0, pin_memory=pin)
        logger.info(f"    WeightedRandomSampler: pos_w={weight_pos:.2f}  neg_w={weight_neg:.2f}")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=pin)

    dev_loader = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=pin)

    model_kwargs = {"num_labels": 2, "ignore_mismatched_sizes": True}
    if "deberta" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.to(device)

    if freeze_layers > 0:
        frozen, encoder = 0, None
        if hasattr(model, "roberta"):
            encoder = model.roberta.encoder.layer
        if encoder is not None:
            for i, layer in enumerate(encoder):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
                    frozen += 1
            logger.info(f"    Froze bottom {frozen} encoder layers (of {len(encoder)})")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay, eps=1e-6)

    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    if loss_mode == "standard":
        loss_fn = lambda logits, labels: F.cross_entropy(logits, labels)
    else:
        _crit   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
        loss_fn = lambda logits, labels: _crit(logits, labels)

    use_amp = False
    scaler  = None

    best_dev_f1    = -1.0
    best_train_loss = float("inf")
    best_threshold = 0.5
    best_metrics   = {}
    best_state     = None
    best_epoch     = 0
    no_improve     = 0

    for epoch in range(1, epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_fn, device, scaler=scaler, use_amp=use_amp)
        if not np.isfinite(train_loss):
            logger.warning("    Training diverged. Stopping early.")
            break
        dev_probs, dev_labels = get_probabilities(model, dev_loader, device, use_amp=use_amp)
        if not np.isfinite(dev_probs).all():
            logger.warning("    Non-finite dev probabilities. Stopping early.")
            break
        threshold = tune_threshold(dev_probs, dev_labels)[0]
        metrics   = evaluate_at_threshold(dev_probs, dev_labels, threshold)
        elapsed   = time.time() - t0

        logger.info(
            f"    Epoch {epoch}/{epochs}  loss={train_loss:.4f}  "
            f"dev F1(PCL)={metrics['f1_pcl']:.4f}  "
            f"P={metrics['precision_pcl']:.4f}  R={metrics['recall_pcl']:.4f}  "
            f"thr={threshold:.2f}  ({elapsed:.0f}s)"
        )
        logger.info(
            f"    prob_stats: mean={dev_probs.mean():.3f}  std={dev_probs.std():.4f}  "
            f"min={dev_probs.min():.3f}  max={dev_probs.max():.3f}"
        )

        f1_improved   = metrics["f1_pcl"] > best_dev_f1
        loss_tiebreak = (
            metrics["f1_pcl"] == best_dev_f1
            and np.isfinite(train_loss)
            and train_loss < best_train_loss
        )
        if f1_improved or loss_tiebreak:
            best_dev_f1     = metrics["f1_pcl"]
            best_train_loss = train_loss
            best_threshold  = threshold
            best_metrics    = metrics
            best_epoch      = epoch
            if f1_improved:
                no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"    Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return best_dev_f1, best_threshold, best_metrics, best_state


# ---------------------------------------------------------------------------
# 9. Train one fixed config and save checkpoint
# ---------------------------------------------------------------------------

def train_fixed_config(cfg, X_train, y_train, X_dev, y_dev, device, output_dir, patience=3):
    """
    Train a single model config and save its checkpoint.

    If the checkpoint directory already exists and contains tokenizer.json,
    training is skipped (resume-friendly for long multi-model runs).

    Returns the best dev F1, or None if skipped.
    """
    ckpt_dir = Path(output_dir) / f"{cfg['name']}{CKPT_SUFFIX}"

    if ckpt_dir.exists() and (ckpt_dir / "tokenizer.json").exists():
        logger.info(f"  [{cfg['name']}] Checkpoint already exists — skipping training.")
        return None

    logger.info(f"\n{'='*62}")
    logger.info(f"  Training: {cfg['name']}  ({cfg['model_name']})")
    logger.info(
        f"  loss={cfg['loss_mode']}  α={cfg['focal_alpha']}  γ={cfg['focal_gamma']}  "
        f"smooth={cfg['label_smoothing']}"
    )
    logger.info(
        f"  lr={cfg['lr']:.1e}  bs={cfg['batch_size']}  "
        f"ep={cfg['epochs']}  seed={cfg['seed']}  max_len={cfg['max_len']}"
    )
    logger.info(f"{'='*62}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    best_f1, best_thr, best_metrics, best_state = run_training_improved(
        model_name          = cfg["model_name"],
        X_train=X_train, y_train=y_train,
        X_dev=X_dev,     y_dev=y_dev,
        tokenizer           = tokenizer,
        lr                  = cfg["lr"],
        batch_size          = cfg["batch_size"],
        epochs              = cfg["epochs"],
        weight_decay        = cfg["weight_decay"],
        max_len             = cfg["max_len"],
        seed                = cfg["seed"],
        device              = device,
        loss_mode           = cfg["loss_mode"],
        focal_gamma         = cfg["focal_gamma"],
        focal_alpha         = cfg["focal_alpha"],
        label_smoothing     = cfg["label_smoothing"],
        use_weighted_sampler= False,   # focal loss alone handles imbalance
        freeze_layers       = 0,
        patience            = patience,
    )

    if best_state is None:
        logger.warning(f"  [{cfg['name']}] Training produced no valid state — skipping save.")
        return None

    logger.info(
        f"  [{cfg['name']}] Best dev F1={best_f1:.4f}  "
        f"P={best_metrics.get('precision_pcl',0):.4f}  "
        f"R={best_metrics.get('recall_pcl',0):.4f}  thr={best_thr:.3f}"
    )

    # Save checkpoint.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=2, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(best_state)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    logger.info(f"  [{cfg['name']}] Checkpoint saved → {ckpt_dir}")

    return best_f1


# ---------------------------------------------------------------------------
# 10. Ensemble evaluation from saved checkpoints
# ---------------------------------------------------------------------------

def run_ensemble_eval(checkpoint_dirs, X_dev, y_dev, device):
    """
    Load each saved checkpoint, get dev probabilities, average them,
    tune threshold, and report individual + ensemble F1.
    """
    y_dev_arr = np.array(y_dev)
    all_probs = []
    per_model = {}

    for ckpt_dir in checkpoint_dirs:
        ckpt_dir = Path(ckpt_dir)
        logger.info(f"  Loading {ckpt_dir.name} ...")
        tokenizer    = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model_kwargs = {"num_labels": 2, "ignore_mismatched_sizes": True}
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_dir), **model_kwargs
        )
        model.to(device).eval()

        dev_ds     = PCLDataset(X_dev, y_dev, tokenizer, max_len=256)
        dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)
        probs, _   = get_probabilities(model, dev_loader, device)
        all_probs.append(probs)

        thr, _ = tune_threshold(probs, y_dev_arr)
        m      = evaluate_at_threshold(probs, y_dev_arr, thr)
        per_model[ckpt_dir.name] = {"f1": m["f1_pcl"], "p": m["precision_pcl"],
                                    "r": m["recall_pcl"], "thr": thr}
        logger.info(
            f"    {ckpt_dir.name:<38}  "
            f"F1={m['f1_pcl']:.4f}  P={m['precision_pcl']:.4f}  "
            f"R={m['recall_pcl']:.4f}  thr={thr:.2f}"
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not all_probs:
        logger.error("No checkpoints loaded for ensemble evaluation.")
        return None

    ens_probs = np.mean(all_probs, axis=0)
    ens_thr, _ = tune_threshold(ens_probs, y_dev_arr)
    ens_m      = evaluate_at_threshold(ens_probs, y_dev_arr, ens_thr)

    logger.info(
        f"\n  ENSEMBLE ({len(all_probs)} models)  "
        f"F1={ens_m['f1_pcl']:.4f}  P={ens_m['precision_pcl']:.4f}  "
        f"R={ens_m['recall_pcl']:.4f}  thr={ens_thr:.2f}"
    )

    return {
        "per_model":          per_model,
        "ensemble_f1":        ens_m["f1_pcl"],
        "ensemble_precision": ens_m["precision_pcl"],
        "ensemble_recall":    ens_m["recall_pcl"],
        "ensemble_threshold": ens_thr,
        "ensemble_metrics":   ens_m,
    }


# ---------------------------------------------------------------------------
# 11. Predict on hidden-label test set
# ---------------------------------------------------------------------------

def run_predict(args, device):
    """
    Load all saved checkpoints from --output_dir, average their PCL
    probabilities, tune threshold on the official dev set, and write
    predictions for the test file to test_predictions.csv.

    Usage:
        python stage4_roberta_ensemble.py --mode predict \\
            --test_file /path/to/task4_test.tsv
    """
    output_dir = Path(args.output_dir)

    candidate_names = [f"{cfg['name']}{CKPT_SUFFIX}" for cfg in CONFIGS]
    checkpoint_dirs = []
    for name in candidate_names:
        ckpt = output_dir / name
        if ckpt.exists() and (ckpt / "tokenizer.json").exists():
            checkpoint_dirs.append(ckpt)
            logger.info(f"  Found checkpoint: {name}")

    if not checkpoint_dirs:
        logger.error(
            f"No checkpoints found in {output_dir}. "
            "Run training first (--mode all, --mode large, or --mode base)."
        )
        return

    logger.info(f"Ensemble size: {len(checkpoint_dirs)} model(s)")

    X_dev, y_dev = load_data(DEV_LABELS_PATH, PCL_TSV_PATH, split_name="dev")
    y_dev_arr    = np.array(y_dev)

    test_texts = read_texts(args.test_file, min_cols=5, skip_lines=0)
    if not test_texts:
        logger.error(f"No texts loaded from {args.test_file}. Check file format.")
        return
    par_ids      = list(test_texts.keys())
    X_test       = [test_texts[pid] for pid in par_ids]
    y_test_dummy = [0] * len(X_test)
    logger.info(f"Test set: {len(X_test)} paragraphs")

    all_dev_probs, all_test_probs = [], []

    for ckpt_dir in checkpoint_dirs:
        logger.info(f"Inferring with {ckpt_dir.name} ...")
        tokenizer    = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model_kwargs = {"num_labels": 2, "ignore_mismatched_sizes": True}
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_dir), **model_kwargs
        )
        model.to(device).eval()

        dev_ds     = PCLDataset(X_dev, y_dev, tokenizer, max_len=256)
        dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)
        dev_probs, _ = get_probabilities(model, dev_loader, device)
        all_dev_probs.append(dev_probs)

        test_ds     = PCLDataset(X_test, y_test_dummy, tokenizer, max_len=256)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        test_probs, _ = get_probabilities(model, test_loader, device)
        all_test_probs.append(test_probs)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ens_dev_probs  = np.mean(all_dev_probs,  axis=0)
    ens_test_probs = np.mean(all_test_probs, axis=0)

    threshold, _ = tune_threshold(ens_dev_probs, y_dev_arr)
    dev_m        = evaluate_at_threshold(ens_dev_probs, y_dev_arr, threshold)
    logger.info(
        f"Ensemble dev  F1={dev_m['f1_pcl']:.4f}  "
        f"P={dev_m['precision_pcl']:.4f}  "
        f"R={dev_m['recall_pcl']:.4f}  thr={threshold:.3f}"
    )

    test_preds = (ens_test_probs >= threshold).astype(int)
    n_pcl      = int(test_preds.sum())
    logger.info(
        f"Test predictions: {n_pcl} PCL, {len(test_preds)-n_pcl} No-PCL "
        f"({100*n_pcl/len(test_preds):.1f}%)"
    )

    pred_path = output_dir / "test_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["par_id", "prediction", "pcl_probability"])
        for pid, pred, prob in zip(par_ids, test_preds, ens_test_probs):
            writer.writerow([pid, int(pred), f"{prob:.6f}"])
    logger.info(f"Predictions saved → {pred_path}")


# ---------------------------------------------------------------------------
# 12. Results comparison table
# ---------------------------------------------------------------------------

def print_comparison(ensemble_result):
    print("\n" + "=" * 72)
    print("  RESULTS COMPARISON")
    print("=" * 72)
    print(f"{'Model':<45} {'F1(PCL)':>8} {'Prec':>8} {'Rec':>8} {'Thr':>6}")
    print("-" * 72)

    for name, r in PRIOR_RESULTS.items():
        print(f"{name + ' dev':<45} {r['dev_f1']:>8.4f} {r['dev_p']:>8.4f} "
              f"{r['dev_r']:>8.4f}")

    if ensemble_result is None:
        print("=" * 72)
        return

    print("-" * 72)
    for name, m in ensemble_result["per_model"].items():
        label = name.replace(CKPT_SUFFIX, "") + " dev"
        print(f"{label:<45} {m['f1']:>8.4f} {m['p']:>8.4f} {m['r']:>8.4f} {m['thr']:>6.2f}")

    print("-" * 72)
    ens = ensemble_result
    print(
        f"{'ENSEMBLE (stage4) dev':<45} {ens['ensemble_f1']:>8.4f} "
        f"{ens['ensemble_precision']:>8.4f} {ens['ensemble_recall']:>8.4f} "
        f"{ens['ensemble_threshold']:>6.2f}"
    )
    print("=" * 72)

    best_prior = max(r["dev_f1"] for r in PRIOR_RESULTS.values())
    delta      = ens["ensemble_f1"] - best_prior
    print(f"  Δ dev F1 vs best prior ({best_prior:.4f}): {delta:+.4f}")
    print()


# ---------------------------------------------------------------------------
# 13. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: multi-RoBERTa ensemble (base + large)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "large", "base", "ensemble", "predict"],
        default="all",
        help=(
            "all=train all 5 models then ensemble; "
            "large=train roberta-large models only; "
            "base=train roberta-base models only; "
            "ensemble=skip training, just evaluate saved checkpoints; "
            "predict=write test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test TSV (required for --mode predict).",
    )
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output_dir", type=str,   default="outputs_stage4")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_versions()
    device = get_device()
    logger.info(f"Device: {device}")

    if args.mode == "predict":
        if args.test_file is None:
            print("ERROR: --test_file is required for --mode predict")
            sys.exit(1)
        run_predict(args, device)
        return

    X_train, y_train = load_data(TRAIN_LABELS_PATH, PCL_TSV_PATH, split_name="train")
    X_dev,   y_dev   = load_data(DEV_LABELS_PATH,   PCL_TSV_PATH, split_name="dev")

    if args.mode == "ensemble":
        # Just evaluate whatever checkpoints already exist.
        candidate_names = [f"{cfg['name']}{CKPT_SUFFIX}" for cfg in CONFIGS]
        checkpoint_dirs = [
            Path(args.output_dir) / n
            for n in candidate_names
            if (Path(args.output_dir) / n / "tokenizer.json").exists()
        ]
        if not checkpoint_dirs:
            print(f"No checkpoints found in {args.output_dir}. Run training first.")
            sys.exit(1)
        result = run_ensemble_eval(checkpoint_dirs, X_dev, y_dev, device)
        print_comparison(result)
        return

    # Training modes: all / large / base
    names_to_run = MODE_CONFIGS[args.mode]
    cfg_map      = {c["name"]: c for c in CONFIGS}

    for name in names_to_run:
        cfg = cfg_map[name]
        train_fixed_config(
            cfg=cfg,
            X_train=X_train, y_train=y_train,
            X_dev=X_dev,     y_dev=y_dev,
            device=device,
            output_dir=args.output_dir,
            patience=args.patience,
        )

    # After training, ensemble all checkpoints that exist (not just this run's).
    all_candidate_names = [f"{cfg['name']}{CKPT_SUFFIX}" for cfg in CONFIGS]
    checkpoint_dirs = [
        Path(args.output_dir) / n
        for n in all_candidate_names
        if (Path(args.output_dir) / n / "tokenizer.json").exists()
    ]

    if checkpoint_dirs:
        logger.info(f"\nEvaluating ensemble of {len(checkpoint_dirs)} model(s)...")
        result = run_ensemble_eval(checkpoint_dirs, X_dev, y_dev, device)

        results_path = Path(args.output_dir) / "stage4_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved → {results_path}")

        print_comparison(result)
    else:
        logger.warning("No checkpoints saved — nothing to ensemble.")


if __name__ == "__main__":
    main()
