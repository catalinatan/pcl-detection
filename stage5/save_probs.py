#!/usr/bin/env python3
"""
save_probs.py — Generate and cache ensemble dev-set probabilities.

Run this once on the machine that has the model checkpoints, then copy
stage5/dev_probs.npy to wherever you run error_analysis.py.

    python stage5/save_probs.py

Outputs: stage5/dev_probs.npy  (shape: [2094], float32)

The probabilities are the ensemble average of:
  hpo_roberta_ce_checkpoint + hpo_roberta_focal_checkpoint
(same two checkpoints used to generate predictions/dev.txt)
"""

import csv
import ast
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_STAGE_DIR = Path(__file__).resolve().parent
_BASE_DIR  = _STAGE_DIR.parent

DEV_LABELS_PATH = _BASE_DIR / "dataset/train/labels/dev_semeval_parids-labels.csv"
PCL_TSV_PATH    = _BASE_DIR / "dataset/train/data/dontpatronizeme_pcl.tsv"
OUT_PATH        = _STAGE_DIR / "dev_probs.npy"

# HPO checkpoints (same ones used for dev.txt)
CKPT_DIR = _BASE_DIR / "stage4" / "outputs_stage4"
DEV_CHECKPOINTS = [
    str(CKPT_DIR / "hpo_roberta_ce_checkpoint"),
    str(CKPT_DIR / "hpo_roberta_focal_checkpoint"),
]

BATCH_SIZE = 32
MAX_LEN    = 256   # matches hpo_roberta_focal max_len (safe upper bound)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data helpers (mirrors stage4_final.py)
# ---------------------------------------------------------------------------

def read_labels(path: Path) -> dict:
    """Returns OrderedDict {par_id: binary_label} preserving CSV row order."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vec = ast.literal_eval(row["label"])
            result[row["par_id"]] = 1 if sum(vec) > 0 else 0
    return result


def read_texts(path: Path) -> dict:
    """Returns {par_id: text} from dontpatronizeme_pcl.tsv (skip 4 header lines)."""
    texts = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            p = line.strip().split("\t")
            if len(p) >= 5:
                texts[p[0]] = p[4]
    return texts


# ---------------------------------------------------------------------------
# Dataset + inference (mirrors stage4_final.py)
# ---------------------------------------------------------------------------

class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = [int(l) for l in labels]
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


@torch.no_grad()
def get_probabilities(model, loader, device) -> np.ndarray:
    model.eval()
    probs_out = []
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs  = F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        probs_out.extend(probs.tolist())
    return np.array(probs_out, dtype=np.float32)


def infer_ensemble(checkpoints: list, texts: list, device: torch.device) -> np.ndarray:
    y_dummy = [0] * len(texts)
    all_probs = []
    for ckpt_dir in checkpoints:
        ckpt_name = Path(ckpt_dir).name
        log.info(f"  Loading {ckpt_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, num_labels=2, ignore_mismatched_sizes=True
        )
        model.to(device).eval()
        loader = DataLoader(
            PCLDataset(texts, y_dummy, tokenizer, max_len=MAX_LEN),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )
        probs = get_probabilities(model, loader, device)
        log.info(f"  {ckpt_name}: mean={probs.mean():.4f}  std={probs.std():.4f}")
        all_probs.append(probs)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.mean(all_probs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Validate checkpoints exist
    for ckpt in DEV_CHECKPOINTS:
        if not (Path(ckpt) / "config.json").exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Run stage4_final.py --mode hpo first."
            )

    # Load dev texts in label-file order (must match predictions/dev.txt row order)
    log.info("Loading dev labels and texts ...")
    labels_map = read_labels(DEV_LABELS_PATH)
    texts_map  = read_texts(PCL_TSV_PATH)

    par_ids = list(labels_map.keys())
    texts   = [texts_map[pid] for pid in par_ids if pid in texts_map]
    log.info(f"  {len(texts)} dev paragraphs (matched in TSV)")

    if len(texts) != len(labels_map):
        raise ValueError(
            f"TSV/CSV mismatch: {len(texts)} texts vs {len(labels_map)} label rows. "
            "Check PCL_TSV_PATH."
        )

    # Run ensemble inference
    log.info("\nRunning ensemble inference ...")
    probs = infer_ensemble(DEV_CHECKPOINTS, texts, device)
    log.info(f"\nEnsemble: mean={probs.mean():.4f}  std={probs.std():.4f}  "
             f"min={probs.min():.4f}  max={probs.max():.4f}")

    # Save
    np.save(OUT_PATH, probs)
    log.info(f"\nSaved {len(probs)} probabilities → {OUT_PATH}")
    log.info("Now run: python stage5/error_analysis.py")


if __name__ == "__main__":
    main()
