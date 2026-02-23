#!/usr/bin/env python3
"""
stage3_improved.py — Additional experiments to improve PCL detection F1.

Builds on stage3.py results:
    Baseline test F1(PCL) = 0.4907   (RoBERTa-base + std CE)
    Novel    test F1(PCL) = 0.5391   (RoBERTa-base + focal loss)

New experiments:
    1. ensemble    — Average probs from baseline + novel checkpoints (zero cost)
    2. deberta     — DeBERTa-v3-base + focal loss (expected biggest gain)
    3. deberta_plus — DeBERTa-v3-base + focal + weighted sampler + label smoothing
                      + freeze bottom layers (kitchen sink)
    4. all         — Run all of the above sequentially

Uses identical data splits (same seed=42, 70/15/15) as stage3.py for
fair comparison.

Usage:
    python stage3_improved.py --mode ensemble
    python stage3_improved.py --mode deberta --n_trials 5 --epochs 5
    python stage3_improved.py --mode deberta_plus --n_trials 5 --epochs 5
    python stage3_improved.py --mode all --n_trials 5 --epochs 5
    python stage3_improved.py --mode all --n_trials 2 --epochs 2          # smoke-test

Outputs → outputs_improved/
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

# Data paths (same as stage3.py).
TRAIN_LABELS_PATH = "train_semeval_parids-labels.csv"
DEV_LABELS_PATH = "dev_semeval_parids-labels.csv"
PCL_TSV_PATH = "dontpatronizeme_pcl.tsv"

# Include 0.00 so we can recover from very low-confidence models instead of
# getting stuck with all-negative predictions and F1=0.
THRESHOLD_RANGE = np.arange(0.00, 1.01, 0.01)

# Prior results for comparison table.
PRIOR_RESULTS = {
    "Baseline (RoBERTa + CE)": {
        "dev_f1": 0.5865, "dev_p": 0.6854, "dev_r": 0.5126, "dev_thr": 0.06,
        "test_f1": 0.4907, "test_p": 0.5464, "test_r": 0.4454, "test_thr": 0.06,
    },
    "Novel (RoBERTa + focal)": {
        "dev_f1": 0.5887, "dev_p": 0.5659, "dev_r": 0.6134, "dev_thr": 0.08,
        "test_f1": 0.5391, "test_p": 0.5036, "test_r": 0.5798, "test_thr": 0.08,
    },
}


# ---------------------------------------------------------------------------
# 2. Utilities (identical to stage3.py for reproducibility)
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


def log_versions() -> dict:
    import sklearn, transformers
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
    }
    logger.info("─── Package versions ───────────────────────────")
    for k, v in versions.items():
        logger.info(f"  {k:<14} {v}")
    logger.info("────────────────────────────────────────────────")
    return versions


# ---------------------------------------------------------------------------
# 3. Data loading (identical to stage3.py)
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
    Read par_id → text from a TSV file.
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
    texts = read_texts(pcl_path)
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
# 4. Dataset
# ---------------------------------------------------------------------------

class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
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
    Alpha-balanced focal loss (Lin et al. 2017).
    Optional label smoothing (Szegedy et al. 2016).
    """

    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        logits = logits.float()  # cast to float32 for numerical stability
        log_softmax = F.log_softmax(logits, dim=-1)
        # log_pt ∈ (-inf, 0];  pt = softmax(logits)[target] ∈ [0, 1]
        log_pt = log_softmax.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()

        if self.label_smoothing > 0:
            # Smooth CE: (1-ε)·(-log pt) + ε·(-mean log p)
            ce_loss = -(
                (1.0 - self.label_smoothing) * log_pt
                + self.label_smoothing * log_softmax.mean(dim=-1)
            )
        else:
            ce_loss = -log_pt

        # Clamp prevents inf when pt ≈ 0 (random-init extreme logits)
        ce_loss = ce_loss.clamp(max=100.0)

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

def train_epoch(model, loader, optimizer, scheduler, loss_fn, device,
                grad_clip=1.0, scaler=None, use_amp=False):
    model.train()
    # When scaler is provided use fp16 (legacy path); otherwise use bf16 (no scaler needed).
    _use_amp = (use_amp or scaler is not None) and device.type == "cuda"
    _amp_dtype = torch.float16 if scaler is not None else torch.bfloat16
    total_loss = 0.0
    n_valid = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=_use_amp, dtype=_amp_dtype):
            logits = model(input_ids=ids, attention_mask=mask).logits
            if not torch.isfinite(logits).all():
                logger.warning("    Non-finite logits detected; skipping batch.")
                continue
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                logger.warning("    Non-finite loss detected; skipping batch.")
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
        n_valid += 1
    if n_valid == 0:
        logger.warning("    All batches had non-finite loss; epoch failed.")
        return float("nan")
    return total_loss / n_valid


@torch.no_grad()
def get_probabilities(model, loader, device, use_amp=False):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
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
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        recall = recall_score(labels, preds, pos_label=1, zero_division=0)
        precision = precision_score(labels, preds, pos_label=1, zero_division=0)
        # Primary objective: F1. Tie-breakers prefer higher recall then precision.
        if (
            f1 > best_f1
            or (f1 == best_f1 and recall > best_recall)
            or (f1 == best_f1 and recall == best_recall and precision > best_precision)
        ):
            best_f1 = f1
            best_t = round(float(t), 3)
            best_recall = recall
            best_precision = precision
    # Safety fallback: if all thresholds collapse to F1=0 but positives exist,
    # force threshold 0.00 to avoid all-negative output.
    if best_f1 <= 0.0 and int(labels.sum()) > 0:
        best_t = 0.0
        preds = (probs >= best_t).astype(int)
        best_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    return best_t, best_f1


def evaluate_at_threshold(probs, labels, threshold):
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
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def freeze_bottom_layers(model, n_freeze=6):
    """
    Freeze the bottom `n_freeze` encoder layers of a transformer model.

    Rationale: Lower layers capture general syntax/morphology; upper layers
    capture task-specific semantics.  Freezing bottom layers reduces
    overfitting on small datasets (5.8K train) and speeds up training.

    Reference: Howard & Ruder 2018 (ULMFiT), Sun et al. 2019 (BERT fine-tuning).
    """
    frozen = 0
    # Works for both RoBERTa (.roberta.encoder.layer) and DeBERTa (.deberta.encoder.layer)
    encoder = None
    if hasattr(model, "roberta"):
        encoder = model.roberta.encoder.layer
    elif hasattr(model, "deberta"):
        encoder = model.deberta.encoder.layer

    if encoder is not None:
        for i, layer in enumerate(encoder):
            if i < n_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                frozen += 1
        logger.info(f"    Froze bottom {frozen} encoder layers (of {len(encoder)})")
    else:
        logger.warning("    Could not find encoder layers to freeze")


# ---------------------------------------------------------------------------
# 7. Experiment A: Ensemble (retrained checkpoints)
# ---------------------------------------------------------------------------

def run_ensemble(
    X_train, y_train,
    X_dev, y_dev,
    device, output_dir, args,
):
    """
    Retrain baseline + novel RoBERTa checkpoints (stage3 recipes) and average
    their PCL probabilities.
    """
    logger.info("\n" + "=" * 62)
    logger.info(" EXPERIMENT: ENSEMBLE (baseline + novel)")
    logger.info("=" * 62)

    results = {}

    # Always retrain so stage3_improved.py is fully standalone.
    resolved_ckpts = {
        "baseline": str(
            retrain_ensemble_checkpoint(
                variant="baseline",
                output_dir=output_dir,
                X_train=X_train, y_train=y_train,
                X_dev=X_dev, y_dev=y_dev,
                seed=args.seed, device=device,
                epochs=max(2, min(4, args.epochs)),
                patience=max(1, min(2, args.patience)),
            )
        ),
        "novel": str(
            retrain_ensemble_checkpoint(
                variant="novel",
                output_dir=output_dir,
                X_train=X_train, y_train=y_train,
                X_dev=X_dev, y_dev=y_dev,
                seed=args.seed, device=device,
                epochs=max(2, min(4, args.epochs)),
                patience=max(1, min(2, args.patience)),
            )
        ),
    }

    for label, ckpt_dir in [("baseline", resolved_ckpts["baseline"]), ("novel", resolved_ckpts["novel"])]:
        # Validate that key files are non-empty (catch corrupt saves).
        for fname in ["tokenizer.json", "model.safetensors"]:
            fpath = Path(ckpt_dir) / fname
            if not fpath.exists():
                # Also accept pytorch_bin format.
                if fname == "model.safetensors" and (Path(ckpt_dir) / "pytorch_model.bin").exists():
                    continue
                logger.error(f"Missing file: {fpath}")
                return None
            if fpath.stat().st_size == 0:
                logger.error(f"Corrupt (0-byte) file: {fpath}")
                return None
        # Quick JSON validity check on tokenizer.json.
        tok_path = Path(ckpt_dir) / "tokenizer.json"
        if tok_path.exists():
            try:
                import json
                with open(tok_path, "r") as f:
                    json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Corrupt tokenizer file: {tok_path} — {e}")
                return None

    # Load both models.
    models = {}
    tokenizers = {}
    for label, ckpt_dir in [("baseline", resolved_ckpts["baseline"]), ("novel", resolved_ckpts["novel"])]:
        logger.info(f"Loading {label} checkpoint from {ckpt_dir}")
        tokenizers[label] = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        model.to(device)
        model.eval()
        models[label] = model

    # Use baseline tokenizer for dataset (both are roberta-base).
    tokenizer = tokenizers["baseline"]

    for split_name, X_split, y_split in [("dev", X_dev, y_dev)]:
        ds = PCLDataset(X_split, y_split, tokenizer, max_len=256)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        # Get probs from each model.
        probs_b, labels = get_probabilities(models["baseline"], loader, device)
        probs_n, _ = get_probabilities(models["novel"], loader, device)

        # Average probs.
        probs_avg = (probs_b + probs_n) / 2.0

        thr, _ = tune_threshold(probs_avg, labels)
        metrics = evaluate_at_threshold(probs_avg, labels, thr)

        logger.info(
            f"  Ensemble {split_name}: F1={metrics['f1_pcl']:.4f}  "
            f"P={metrics['precision_pcl']:.4f}  R={metrics['recall_pcl']:.4f}  thr={thr}"
        )
        results[f"{split_name}_metrics"] = metrics
        results[f"{split_name}_threshold"] = thr

    # Clean up.
    for m in models.values():
        del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def retrain_ensemble_checkpoint(
    variant,
    output_dir,
    X_train, y_train,
    X_dev, y_dev,
    seed,
    device,
    epochs=3,
    patience=2,
):
    """
    Retrain RoBERTa checkpoints for ensemble mode using stage3 recipes:
    - baseline: standard cross-entropy
    - novel: focal loss
    """
    out = Path(output_dir) / f"retrained_{variant}_checkpoint"
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Retraining `{variant}` checkpoint at {out}")

    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if variant == "baseline":
        loss_mode = "standard"
        focal_gamma, focal_alpha, label_smoothing = 2.0, 0.25, 0.0
    else:
        # alpha=0.75 upweights the minority PCL class (9.5% positive rate).
        # This gives the novel member higher recall vs the baseline's higher
        # precision — creating the complementary diversity that makes ensembling
        # worthwhile. alpha=0.25 (original) would make both members conservative.
        loss_mode = "focal"
        focal_gamma, focal_alpha, label_smoothing = 2.0, 0.75, 0.0

    _, _, _, best_state = run_training_improved(
        model_name=model_name,
        X_train=X_train, y_train=y_train,
        X_dev=X_dev, y_dev=y_dev,
        tokenizer=tokenizer,
        lr=2e-5,
        batch_size=16,
        epochs=epochs,
        weight_decay=0.01,
        max_len=256,
        seed=seed,
        device=device,
        loss_mode=loss_mode,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        label_smoothing=label_smoothing,
        use_weighted_sampler=False,
        freeze_layers=0,
        patience=patience,
    )
    if best_state is None:
        raise RuntimeError(f"Failed to retrain `{variant}` checkpoint.")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(best_state)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


# ---------------------------------------------------------------------------
# 8. Experiment B & C: DeBERTa training
# ---------------------------------------------------------------------------

def run_training_improved(
    model_name,
    X_train, y_train,
    X_dev, y_dev,
    tokenizer,
    *,
    lr, batch_size, epochs, weight_decay, max_len,
    seed, device,
    loss_mode="focal",   # "standard" | "focal"
    focal_gamma=2.0, focal_alpha=0.25,
    label_smoothing=0.0,
    use_weighted_sampler=False,
    freeze_layers=0,
    patience=3,
):
    """
    Enhanced training loop with support for:
    - Focal loss + label smoothing
    - WeightedRandomSampler (data-level class balancing)
    - Layer freezing (reduce overfitting)
    """
    set_seed(seed)

    train_ds = PCLDataset(X_train, y_train, tokenizer, max_len)
    dev_ds = PCLDataset(X_dev, y_dev, tokenizer, max_len)

    pin = device.type == "cuda"

    # --- Weighted sampler: oversample PCL so each batch is ~50/50 ---
    if use_weighted_sampler:
        y_arr = np.array(y_train)
        n_pos = y_arr.sum()
        n_neg = len(y_arr) - n_pos
        weight_pos = len(y_arr) / (2.0 * n_pos)
        weight_neg = len(y_arr) / (2.0 * n_neg)
        sample_weights = np.where(y_arr == 1, weight_pos, weight_neg)
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=pin,
        )
        logger.info(f"    WeightedRandomSampler: pos_w={weight_pos:.2f}  neg_w={weight_neg:.2f}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=pin,
        )

    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    # Prefer eager attention for DeBERTa on CUDA to avoid occasional non-finite
    # logits with newer torch/transformers kernel combinations.
    model_kwargs = {
        "num_labels": 2,
        "ignore_mismatched_sizes": True,
    }
    if "deberta" in model_name.lower():
        model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_kwargs
    )
    model.to(device)

    # --- Layer freezing ---
    if freeze_layers > 0:
        freeze_bottom_layers(model, n_freeze=freeze_layers)

    # Only optimise non-frozen parameters.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(0.15 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Keep stage3 baseline behavior available for retraining.
    if loss_mode == "standard":
        loss_fn = lambda logits, labels: F.cross_entropy(logits, labels)
    else:
        _criterion = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        loss_fn = lambda logits, labels: _criterion(logits, labels)

    # Keep training in full precision for stability. In practice, bf16 autocast
    # can still produce non-finite loss with some DeBERTa/hyperparameter combos.
    use_amp = False
    scaler = None  # bf16 autocast does not need gradient scaling

    best_dev_f1 = -1.0   # -1 so even F1=0.0 on epoch 1 triggers a save
    best_threshold = 0.5
    best_metrics = {}
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 loss_fn, device, scaler=scaler, use_amp=use_amp)
        if not np.isfinite(train_loss):
            logger.warning("    Training diverged (non-finite loss). Stopping this trial early.")
            break
        dev_probs, dev_labels = get_probabilities(model, dev_loader, device,
                                                  use_amp=use_amp)
        if not np.isfinite(dev_probs).all():
            logger.warning("    Non-finite dev probabilities. Stopping this trial early.")
            break
        threshold, _ = tune_threshold(dev_probs, dev_labels)
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


def run_deberta_experiment(
    mode_label,
    model_name,
    X_train, y_train,
    X_dev, y_dev,
    device, args, output_dir,
    use_weighted_sampler=False,
    label_smoothing=0.0,
    freeze_layers=0,
):
    """
    Run DeBERTa-based experiments with hyperparameter search.
    """
    logger.info(f"\n{'='*62}")
    logger.info(f" EXPERIMENT: {mode_label.upper()}")
    logger.info(f" Model: {model_name}")
    logger.info(f" Extras: sampler={use_weighted_sampler}  "
                f"label_smooth={label_smoothing}  freeze={freeze_layers}")
    logger.info(f" Trials: {args.n_trials}  |  Max epochs: {args.epochs}")
    logger.info(f"{'='*62}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    best_holder = {
        "f1": 0.0, "threshold": 0.5, "metrics": {},
        "state": None, "params": {}, "trial_id": -1,
    }
    tuning_rows = []

    rng = np.random.RandomState(args.seed)
    for trial_id in range(args.n_trials):
        # PCL is ~9.5% of data, so alpha must upweight the positive class.
        # alpha=0.25 was wrong (downweights minority); correct range is 0.65–0.85.
        # Batch size 4 causes noisy gradients and NaN loss; minimum is now 8.
        params = {
            "lr": float(np.exp(rng.uniform(np.log(8e-6), np.log(3e-5)))),
            "batch_size": int(rng.choice([8, 16])),
            "epochs": int(rng.randint(min(2, args.epochs), args.epochs + 1)),
            "weight_decay": float(rng.uniform(0.0, 0.05)),
            "max_len": int(rng.choice([128, 256])),
            "focal_gamma": float(rng.uniform(0.5, 2.0)),
            "focal_alpha": float(rng.uniform(0.65, 0.85)),
        }

        logger.info(
            f"\n[{mode_label.upper()} Trial {trial_id}/{args.n_trials-1}] "
            f"lr={params['lr']:.2e}  bs={params['batch_size']}  "
            f"ep={params['epochs']}  wd={params['weight_decay']:.3f}  "
            f"max_len={params['max_len']}  "
            f"γ={params['focal_gamma']:.2f}  α={params['focal_alpha']:.2f}"
        )

        dev_f1, threshold, metrics, state = run_training_improved(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_dev=X_dev, y_dev=y_dev,
            tokenizer=tokenizer,
            lr=params["lr"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            weight_decay=params["weight_decay"],
            max_len=params["max_len"],
            seed=args.seed,
            device=device,
            focal_gamma=params["focal_gamma"],
            focal_alpha=params["focal_alpha"],
            label_smoothing=label_smoothing,
            use_weighted_sampler=use_weighted_sampler,
            freeze_layers=freeze_layers,
            patience=args.patience,
        )

        tuning_rows.append({
            "mode": mode_label,
            "trial_id": trial_id,
            **params,
            "label_smoothing": label_smoothing,
            "weighted_sampler": use_weighted_sampler,
            "freeze_layers": freeze_layers,
            "best_threshold": threshold,
            "dev_f1_pcl": metrics.get("f1_pcl", 0.0),
            "dev_precision": metrics.get("precision_pcl", 0.0),
            "dev_recall": metrics.get("recall_pcl", 0.0),
        })

        if dev_f1 > best_holder["f1"]:
            best_holder.update({
                "f1": dev_f1, "threshold": threshold,
                "metrics": metrics, "state": state,
                "params": params, "trial_id": trial_id,
            })

    # --- Evaluate on official dev with best model ---
    if best_holder["state"] is None:
        logger.warning(f"[{mode_label}] No successful trials.")
        return None, tuning_rows

    logger.info(
        f"\n[{mode_label.upper()}] Best dev F1(PCL)={best_holder['f1']:.4f}  "
        f"thr={best_holder['threshold']:.3f}  "
        f"P={best_holder['metrics'].get('precision_pcl',0):.4f}  "
        f"R={best_holder['metrics'].get('recall_pcl',0):.4f}"
    )

    # Rebuild model, load best weights, eval on test.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(best_holder["state"])
    model.to(device)

    # Save checkpoint.
    ckpt_dir = str(Path(output_dir) / f"{mode_label}_best_checkpoint")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info(f"Checkpoint saved → {ckpt_dir}")

    # Dev evaluation.
    max_len = int(best_holder["params"].get("max_len", 256))
    dev_ds = PCLDataset(X_dev, y_dev, tokenizer, max_len)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)
    use_amp = device.type == "cuda"
    dev_probs, dev_labels = get_probabilities(model, dev_loader, device, use_amp=use_amp)
    dev_eval_metrics = evaluate_at_threshold(dev_probs, dev_labels, best_holder["threshold"])

    # Also check oracle dev threshold.
    oracle_thr, oracle_f1 = tune_threshold(dev_probs, dev_labels)
    dev_eval_metrics["dev_oracle_threshold"] = oracle_thr
    dev_eval_metrics["dev_oracle_f1"] = oracle_f1

    logger.info(
        f"[{mode_label.upper()}] Dev(eval) F1(PCL)={dev_eval_metrics['f1_pcl']:.4f}  "
        f"P={dev_eval_metrics['precision_pcl']:.4f}  "
        f"R={dev_eval_metrics['recall_pcl']:.4f}"
    )

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results = {
        "mode": mode_label,
        "model_name": model_name,
        "best_hyperparams": best_holder["params"],
        "best_threshold": best_holder["threshold"],
        "best_trial_id": best_holder["trial_id"],
        "dev_metrics": best_holder["metrics"],
        "dev_eval_metrics": dev_eval_metrics,
        "extras": {
            "label_smoothing": label_smoothing,
            "weighted_sampler": use_weighted_sampler,
            "freeze_layers": freeze_layers,
        },
    }
    return results, tuning_rows


# ---------------------------------------------------------------------------
# 9. Results comparison
# ---------------------------------------------------------------------------

def print_comparison(all_results):
    """Print a comparison table including prior stage3.py results."""
    print("\n" + "=" * 80)
    print("  FULL RESULTS COMPARISON (prior + new experiments)")
    print("=" * 80)
    print(f"{'Model':<42} {'F1(PCL)':>8} {'Prec':>8} {'Rec':>8} {'Thr':>6}")
    print("-" * 80)

    # Prior results.
    for name, r in PRIOR_RESULTS.items():
        print(f"{name + ' dev':<42} {r['dev_f1']:>8.4f} {r['dev_p']:>8.4f} "
              f"{r['dev_r']:>8.4f} {r['dev_thr']:>6.2f}")
        print(f"{name + ' test':<42} {r['test_f1']:>8.4f} {r['test_p']:>8.4f} "
              f"{r['test_r']:>8.4f} {r['test_thr']:>6.2f}")

    print("-" * 80)

    # New results.
    for name, res in all_results.items():
        if res is None:
            continue
        for split in ["dev"]:
            m = res.get(f"{split}_metrics", {})
            thr = res.get(f"{split}_threshold", res.get("best_threshold", 0))
            if not m and split == "dev":
                m = res.get("dev_eval_metrics", {})
            if m:
                print(
                    f"{name + ' ' + split:<42} {m.get('f1_pcl',0):>8.4f} "
                    f"{m.get('precision_pcl',0):>8.4f} "
                    f"{m.get('recall_pcl',0):>8.4f} {thr:>6.2f}"
                )

    print("=" * 80)

    # Delta vs best prior dev score (since this script now evaluates on dev only).
    best_prior_dev = max(r["dev_f1"] for r in PRIOR_RESULTS.values())
    for name, res in all_results.items():
        if res is None:
            continue
        dev_m = res.get("dev_metrics", {}) or res.get("dev_eval_metrics", {})
        if dev_m:
            delta = dev_m.get("f1_pcl", 0) - best_prior_dev
            print(f"  Δ dev F1 ({name} vs best prior dev): {delta:+.4f}")

    print()


# ---------------------------------------------------------------------------
# 10. Ensemble predict (for hidden-label test set submission)
# ---------------------------------------------------------------------------

def run_predict(args, device):
    """
    Load all saved checkpoints from --output_dir, average their PCL
    probabilities, tune the decision threshold on the official dev set,
    and write predictions for the test file to test_predictions.csv.

    Usage:
        python stage3_improved.py --mode predict --test_file /path/to/test.tsv
    """
    output_dir = Path(args.output_dir)

    # Checkpoints saved by each experiment, in priority order.
    candidate_names = [
        "deberta_focal_best_checkpoint",
        "deberta_plus_best_checkpoint",
        "retrained_novel_checkpoint",
        "retrained_baseline_checkpoint",
    ]
    checkpoint_dirs = []
    for name in candidate_names:
        ckpt = output_dir / name
        if ckpt.exists() and (ckpt / "tokenizer.json").exists():
            checkpoint_dirs.append(ckpt)
            logger.info(f"  Found checkpoint: {name}")

    if not checkpoint_dirs:
        logger.error(
            f"No checkpoints found in {output_dir}. "
            "Run training first (--mode all or individual modes)."
        )
        return

    logger.info(f"Ensemble size: {len(checkpoint_dirs)} model(s)")

    # Load dev set for threshold tuning.
    X_dev, y_dev = load_data(DEV_LABELS_PATH, PCL_TSV_PATH, split_name="dev")
    y_dev_arr = np.array(y_dev)

    # Load test set — no 4-line header, no label column.
    test_texts = read_texts(args.test_file, min_cols=5, skip_lines=0)
    if not test_texts:
        logger.error(f"No texts loaded from {args.test_file}. Check file format.")
        return
    par_ids = list(test_texts.keys())
    X_test = [test_texts[pid] for pid in par_ids]
    y_test_dummy = [0] * len(X_test)
    logger.info(f"Test set: {len(X_test)} paragraphs")

    all_dev_probs, all_test_probs = [], []

    for ckpt_dir in checkpoint_dirs:
        logger.info(f"Inferring with {ckpt_dir.name} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        model_kwargs = {"num_labels": 2, "ignore_mismatched_sizes": True}
        if "deberta" in ckpt_dir.name.lower():
            model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ckpt_dir), **model_kwargs
        )
        model.to(device).eval()

        max_len = 256  # safe default; all checkpoints were trained with ≤256

        dev_ds = PCLDataset(X_dev, y_dev, tokenizer, max_len)
        dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)
        dev_probs, _ = get_probabilities(model, dev_loader, device)
        all_dev_probs.append(dev_probs)

        test_ds = PCLDataset(X_test, y_test_dummy, tokenizer, max_len)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
        test_probs, _ = get_probabilities(model, test_loader, device)
        all_test_probs.append(test_probs)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Average probabilities across all models.
    ens_dev_probs = np.mean(all_dev_probs, axis=0)
    ens_test_probs = np.mean(all_test_probs, axis=0)

    # Tune threshold on ensemble dev probabilities.
    threshold, _ = tune_threshold(ens_dev_probs, y_dev_arr)
    dev_metrics = evaluate_at_threshold(ens_dev_probs, y_dev_arr, threshold)
    logger.info(
        f"Ensemble dev  F1={dev_metrics['f1_pcl']:.4f}  "
        f"P={dev_metrics['precision_pcl']:.4f}  "
        f"R={dev_metrics['recall_pcl']:.4f}  thr={threshold:.3f}"
    )

    # Apply threshold → binary predictions.
    test_preds = (ens_test_probs >= threshold).astype(int)
    n_pcl = int(test_preds.sum())
    logger.info(f"Test predictions: {n_pcl} PCL, {len(test_preds)-n_pcl} No-PCL "
                f"({100*n_pcl/len(test_preds):.1f}%)")

    # Save CSV: par_id, prediction, pcl_probability
    pred_path = output_dir / "test_predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["par_id", "prediction", "pcl_probability"])
        for pid, pred, prob in zip(par_ids, test_preds, ens_test_probs):
            writer.writerow([pid, int(pred), f"{prob:.6f}"])
    logger.info(f"Predictions saved → {pred_path}")


# ---------------------------------------------------------------------------
# 11. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Improved: additional experiments for PCL detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ensemble", "deberta", "deberta_plus", "all", "predict"],
        default="all",
        help="Which experiment(s) to run. Use 'predict' to generate test predictions.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test TSV (required for --mode predict). Labels not needed.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_improved")
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

    # Use official split files directly.
    X_train, y_train = load_data(TRAIN_LABELS_PATH, PCL_TSV_PATH, split_name="train")
    X_dev, y_dev = load_data(DEV_LABELS_PATH, PCL_TSV_PATH, split_name="dev")

    modes = {
        "ensemble": ["ensemble"],
        "deberta": ["deberta"],
        "deberta_plus": ["deberta_plus"],
        "all": ["ensemble", "deberta", "deberta_plus"],
    }[args.mode]

    all_results = {}
    all_tuning_rows = []

    for mode in modes:
        if mode == "ensemble":
            res = run_ensemble(
                X_train=X_train, y_train=y_train,
                X_dev=X_dev, y_dev=y_dev,
                device=device,
                output_dir=args.output_dir,
                args=args,
            )
            all_results["Ensemble (base+novel)"] = res

        elif mode == "deberta":
            # DeBERTa-v3-base + focal loss (same recipe as novel but better encoder).
            res, rows = run_deberta_experiment(
                mode_label="deberta_focal",
                model_name="microsoft/deberta-v3-base",
                X_train=X_train, y_train=y_train,
                X_dev=X_dev, y_dev=y_dev,
                device=device, args=args,
                output_dir=args.output_dir,
                use_weighted_sampler=False,
                label_smoothing=0.0,
                freeze_layers=0,
            )
            all_results["DeBERTa + focal"] = res
            all_tuning_rows.extend(rows)

        elif mode == "deberta_plus":
            # DeBERTa + focal + weighted sampler + label smoothing + freeze bottom 6 layers.
            res, rows = run_deberta_experiment(
                mode_label="deberta_plus",
                model_name="microsoft/deberta-v3-base",
                X_train=X_train, y_train=y_train,
                X_dev=X_dev, y_dev=y_dev,
                device=device, args=args,
                output_dir=args.output_dir,
                use_weighted_sampler=True,
                label_smoothing=0.05,
                freeze_layers=6,
            )
            all_results["DeBERTa + focal + extras"] = res
            all_tuning_rows.extend(rows)

    # Save tuning log.
    if all_tuning_rows:
        log_path = Path(args.output_dir) / "improved_tuning_log.csv"
        fieldnames = list(all_tuning_rows[0].keys())
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_tuning_rows)
        logger.info(f"Tuning log saved → {log_path}")

    # Save results JSON.
    results_path = Path(args.output_dir) / "improved_results.json"
    # Convert for JSON serialisation.
    serialisable = {}
    for k, v in all_results.items():
        if v is not None:
            serialisable[k] = v
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # Print comparison.
    print_comparison(all_results)


if __name__ == "__main__":
    main()
