"""
Exercise 5.2: Local Evaluation — Error Analysis
================================================
Generates all tables and figures for the error analysis write-up.

Sections
--------
1. Overall metrics + confusion matrix         (binary preds)
2. Category-level miss rates                  (binary preds)
3. Annotator agreement vs. errors             (binary preds)
4. FP / FN example tables                     (binary preds)
5. Model comparison bar chart                 (comparison_results.json)
6. Precision-Recall curve + AUC-PR            (needs dev_probs.npy)
7. Threshold tuning curve (F1 vs threshold)   (needs dev_probs.npy)
8. Calibration curve                          (needs dev_probs.npy)
9. Length performance curve                   (binary preds)

Generating dev_probs.npy (required for sections 6-8)
-----------------------------------------------------
Run on the machine that has the model checkpoints:

    python stage5/save_probs.py

This saves stage5/dev_probs.npy which is then read by this script.

Usage (from repo root):
    python stage5/error_analysis.py
"""

import ast
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_STAGE_DIR = Path(__file__).resolve().parent
_BASE_DIR  = _STAGE_DIR.parent

PREDICTIONS_PATH = _BASE_DIR / "predictions" / "dev.txt"
DEV_LABELS_PATH  = _BASE_DIR / "dataset/train/labels/dev_semeval_parids-labels.csv"
PCL_TSV_PATH     = _BASE_DIR / "dataset/train/data/dontpatronizeme_pcl.tsv"
COMPARISON_JSON  = _BASE_DIR / "stage4" / "outputs_stage4" / "comparison_results.json"
PROBS_CACHE      = _STAGE_DIR / "dev_probs.npy"
OUT_DIR          = _STAGE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth():
    pids, labels = [], []
    with open(DEV_LABELS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vec = ast.literal_eval(row["label"])
            pids.append(row["par_id"])
            labels.append(1 if sum(vec) > 0 else 0)
    return pids, labels


def load_predictions():
    with open(PREDICTIONS_PATH, encoding="utf-8") as f:
        return [int(l.strip()) for l in f if l.strip()]


def load_tsv():
    """Returns {par_id: (category, text, pcl_score)}."""
    entries = {}
    with open(PCL_TSV_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            p = line.strip().split("\t")
            if len(p) >= 6:
                entries[p[0]] = (p[2], p[4], int(p[5]) if p[5].isdigit() else -1)
    return entries


# ---------------------------------------------------------------------------
# Section 1 — Overall metrics + confusion matrix
# ---------------------------------------------------------------------------

def section1_overall(gt_labels, preds):
    tp = sum(1 for g, p in zip(gt_labels, preds) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gt_labels, preds) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gt_labels, preds) if g == 1 and p == 0)
    tn = sum(1 for g, p in zip(gt_labels, preds) if g == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    n    = len(gt_labels)

    print("=" * 55)
    print("SECTION 1 — Overall Performance on Official Dev Set")
    print("=" * 55)
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  GT PCL rate  : {sum(gt_labels)/n:.3f}")
    print(f"  Pred PCL rate: {sum(preds)/n:.3f}")

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No-PCL", "Pred: PCL"])
    ax.set_yticklabels(["GT: No-PCL", "GT: PCL"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion Matrix  (F1={f1:.3f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = OUT_DIR / "fig1_confusion_matrix.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Section 2 — Category-level miss rates
# ---------------------------------------------------------------------------

def section2_category(gt_pids, gt_labels, preds, tsv):
    cat_total, cat_tp, cat_fn = Counter(), Counter(), Counter()
    for pid, g, p in zip(gt_pids, gt_labels, preds):
        if pid not in tsv or g != 1:
            continue
        cat = tsv[pid][0]
        cat_total[cat] += 1
        (cat_tp if p == 1 else cat_fn)[cat] += 1

    print("\n" + "=" * 55)
    print("SECTION 2 — Category-level Miss Rates")
    print("=" * 55)
    print(f"  {'Category':<20} {'Total':>6} {'TP':>5} {'FN':>5} {'Miss%':>7}")
    print("  " + "-" * 45)
    rows = sorted(cat_total, key=lambda c: cat_fn[c] / cat_total[c], reverse=True)
    for cat in rows:
        tot = cat_total[cat]
        print(f"  {cat:<20} {tot:>6} {cat_tp[cat]:>5} {cat_fn[cat]:>5} "
              f"{100*cat_fn[cat]/tot:>6.0f}%")

    miss_pcts = [100 * cat_fn[c] / cat_total[c] for c in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(rows[::-1], miss_pcts[::-1], color="salmon")
    ax.set_xlabel("Miss rate (%)")
    ax.set_title("False Negative Miss Rate by Vulnerable Group Category")
    ax.axvline(50, color="grey", linestyle="--", linewidth=0.8)
    for bar, pct in zip(bars, miss_pcts[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "fig2_category_miss_rate.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Section 3 — Annotator agreement vs. errors
# ---------------------------------------------------------------------------

def section3_annotator(gt_pids, gt_labels, preds, tsv):
    fn_scores = [tsv[p][2] for p, g, pr in zip(gt_pids, gt_labels, preds)
                 if g == 1 and pr == 0 and p in tsv and tsv[p][2] >= 0]
    tp_scores = [tsv[p][2] for p, g, pr in zip(gt_pids, gt_labels, preds)
                 if g == 1 and pr == 1 and p in tsv and tsv[p][2] >= 0]

    print("\n" + "=" * 55)
    print("SECTION 3 — Annotator Agreement Score vs. Errors")
    print("=" * 55)
    print(f"  FN avg score : {statistics.mean(fn_scores):.2f}  "
          f"dist={sorted(Counter(fn_scores).items())}")
    print(f"  TP avg score : {statistics.mean(tp_scores):.2f}  "
          f"dist={sorted(Counter(tp_scores).items())}")

    scores    = [2, 3, 4]
    fn_counts = [fn_scores.count(s) for s in scores]
    tp_counts = [tp_scores.count(s) for s in scores]
    x = np.arange(len(scores)); w = 0.35
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - w/2, tp_counts, w, label="TP (caught)", color="steelblue")
    ax.bar(x + w/2, fn_counts, w, label="FN (missed)", color="salmon")
    ax.set_xticks(x); ax.set_xticklabels([f"Score={s}" for s in scores])
    ax.set_ylabel("Count")
    ax.set_title("Annotator Agreement Score:\nCaught vs. Missed PCL")
    ax.legend()
    plt.tight_layout()
    path = OUT_DIR / "fig3_annotator_score.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Section 4 — FP / FN example tables
# ---------------------------------------------------------------------------

def section4_examples(gt_pids, gt_labels, preds, tsv, n=6):
    fp_cases = [(pid, *tsv[pid]) for pid, g, p in zip(gt_pids, gt_labels, preds)
                if g == 0 and p == 1 and pid in tsv]
    fn_cases = [(pid, *tsv[pid]) for pid, g, p in zip(gt_pids, gt_labels, preds)
                if g == 1 and p == 0 and pid in tsv]

    print("\n" + "=" * 55)
    print(f"SECTION 4 — Sample Errors  (FP={len(fp_cases)}  FN={len(fn_cases)})")
    print("=" * 55)
    print(f"\n  FALSE POSITIVES — top {n}:")
    for pid, cat, txt, sc in fp_cases[:n]:
        print(f"    [{cat}] {txt[:200]}\n")
    print(f"  FALSE NEGATIVES — top {n}:")
    for pid, cat, txt, sc in fn_cases[:n]:
        print(f"    [{cat}] score={sc} | {txt[:200]}\n")

    for cases, name in [(fp_cases, "fp_cases.csv"), (fn_cases, "fn_cases.csv")]:
        path = OUT_DIR / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["par_id", "category", "pcl_score", "text"])
            for pid, cat, txt, sc in cases:
                w.writerow([pid, cat, sc, txt])
        print(f"  → {path.name}  ({len(cases)} rows)")


# ---------------------------------------------------------------------------
# Section 5 — Model comparison chart
# ---------------------------------------------------------------------------

def section5_model_comparison():
    if not COMPARISON_JSON.exists():
        print("\nSECTION 5 — comparison_results.json not found, skipping.")
        return

    with open(COMPARISON_JSON) as f:
        cmp = json.load(f)

    print("\n" + "=" * 55)
    print("SECTION 5 — Model Comparison")
    print("=" * 55)
    print(f"  {'Model':<20} {'Dev F1':>8} {'Precision':>10} {'Recall':>8} {'Thr':>6}")
    print("  " + "-" * 55)
    for name in ["roberta_ce", "roberta_focal", "deberta", "ensemble"]:
        if name not in cmp:
            continue
        r = cmp[name]; m = r["metrics"]
        marker = " ★" if name == cmp.get("_best_approach") else ""
        print(f"  {name+marker:<20} {r['official_dev_f1']:>8.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{r['official_dev_thr']:>6.2f}")

    labels = ["RoBERTa-CE", "RoBERTa-Focal", "DeBERTa", "Ensemble"]
    keys   = ["roberta_ce", "roberta_focal", "deberta", "ensemble"]
    f1s    = [cmp[k]["official_dev_f1"] for k in keys]
    precs  = [cmp[k]["metrics"]["precision"] for k in keys]
    recs   = [cmp[k]["metrics"]["recall"] for k in keys]

    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, f1s,   w, label="F1",       color="steelblue")
    ax.bar(x,     precs, w, label="Precision", color="seagreen")
    ax.bar(x + w, recs,  w, label="Recall",    color="salmon")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Official Dev Set")
    ax.legend()
    ax.axhline(0.6, color="grey", linestyle="--", linewidth=0.8, label="F1=0.6")
    for i, (f, p, r) in enumerate(zip(f1s, precs, recs)):
        ax.text(i - w, f + 0.02, f"{f:.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    path = OUT_DIR / "fig4_model_comparison.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Probability loading (required for sections 6-8)
# ---------------------------------------------------------------------------

def load_probs():
    """Load cached dev probabilities. Returns None if not available."""
    if PROBS_CACHE.exists():
        probs = np.load(PROBS_CACHE)
        print(f"  Loaded dev_probs.npy  ({len(probs)} values)")
        return probs
    print(f"  dev_probs.npy not found — skipping sections 6-8.")
    print(f"  To generate it, run: python stage5/save_probs.py")
    return None


# ---------------------------------------------------------------------------
# Section 6 — Precision-Recall Curve
# ---------------------------------------------------------------------------

def section6_pr_curve(gt_labels, probs):
    from sklearn.metrics import precision_recall_curve, auc

    precision, recall, thresholds = precision_recall_curve(gt_labels, probs)
    auc_pr  = auc(recall, precision)
    baseline = sum(gt_labels) / len(gt_labels)   # random classifier baseline

    print("\n" + "=" * 55)
    print("SECTION 6 — Precision-Recall Curve")
    print("=" * 55)
    print(f"  AUC-PR   : {auc_pr:.4f}")
    print(f"  Baseline : {baseline:.4f}  (random classifier for {baseline*100:.1f}% positive rate)")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="steelblue", lw=2,
            label=f"Ensemble (AUC-PR = {auc_pr:.3f})")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Positive Class = PCL)")
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path = OUT_DIR / "fig5_pr_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Section 7 — Threshold Tuning Curve
# ---------------------------------------------------------------------------

def section7_threshold_curve(gt_labels, probs):
    gt = np.array(gt_labels)
    thresholds = np.linspace(0.0, 1.0, 201)
    f1s, precs, recs = [], [], []

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tp = ((preds == 1) & (gt == 1)).sum()
        fp = ((preds == 1) & (gt == 0)).sum()
        fn = ((preds == 0) & (gt == 1)).sum()
        p  = tp / (tp + fp) if (tp + fp) else 0
        r  = tp / (tp + fn) if (tp + fn) else 0
        f  = 2 * p * r / (p + r) if (p + r) else 0
        f1s.append(f); precs.append(p); recs.append(r)

    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]
    best_f1  = f1s[best_idx]

    print("\n" + "=" * 55)
    print("SECTION 7 — Threshold Tuning Curve")
    print("=" * 55)
    print(f"  Best threshold : {best_thr:.2f}")
    print(f"  Best F1        : {best_f1:.4f}")
    print(f"  At τ=0.50      : F1={f1s[100]:.4f}  P={precs[100]:.4f}  R={recs[100]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1s,   color="steelblue", lw=2, label="F1")
    ax.plot(thresholds, precs, color="seagreen",  lw=1.5, linestyle="--", label="Precision")
    ax.plot(thresholds, recs,  color="salmon",    lw=1.5, linestyle="--", label="Recall")
    ax.axvline(best_thr, color="steelblue", linestyle=":", lw=1.5,
               label=f"Best τ={best_thr:.2f} (F1={best_f1:.3f})")
    ax.axvline(0.5, color="grey", linestyle=":", lw=1, label="Default τ=0.5")
    ax.set_xlabel("Decision Threshold (τ)")
    ax.set_ylabel("Score")
    ax.set_title("F1 / Precision / Recall vs. Decision Threshold")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    path = OUT_DIR / "fig6_threshold_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Section 8 — Calibration Curve
# ---------------------------------------------------------------------------

def section8_calibration(gt_labels, probs):
    from sklearn.calibration import calibration_curve

    fraction_pos, mean_pred = calibration_curve(gt_labels, probs, n_bins=10)

    print("\n" + "=" * 55)
    print("SECTION 8 — Calibration Curve")
    print("=" * 55)
    print(f"  {'Pred prob bin':>15}  {'True PCL fraction':>18}")
    for mp, fp in zip(mean_pred, fraction_pos):
        bar = "█" * int(fp * 20)
        print(f"  {mp:>15.3f}  {fp:>18.3f}  {bar}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, fraction_pos, "s-", color="steelblue", lw=2,
            label="Ensemble (RoBERTa-CE + Focal)")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("True PCL Fraction")
    ax.set_title("Calibration Curve\n(below diagonal = underconfident, above = overconfident)")
    ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    path = OUT_DIR / "fig7_calibration.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Section 9 — Length Performance Curve
# ---------------------------------------------------------------------------

def section9_length_curve(gt_pids, gt_labels, preds, tsv):
    # Bucket texts by word count
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120), (120, 300)]
    labels  = [f"{lo}-{hi}" for lo, hi in buckets]

    bucket_f1, bucket_n = [], []
    for lo, hi in buckets:
        gt_b, pr_b = [], []
        for pid, g, p in zip(gt_pids, gt_labels, preds):
            if pid not in tsv:
                continue
            wc = len(tsv[pid][1].split())
            if lo <= wc < hi:
                gt_b.append(g); pr_b.append(p)
        if not gt_b:
            bucket_f1.append(0); bucket_n.append(0); continue
        gt_b = np.array(gt_b); pr_b = np.array(pr_b)
        tp = ((pr_b == 1) & (gt_b == 1)).sum()
        fp = ((pr_b == 1) & (gt_b == 0)).sum()
        fn = ((pr_b == 0) & (gt_b == 1)).sum()
        pr = tp / (tp + fp) if (tp + fp) else 0
        re = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * pr * re / (pr + re) if (pr + re) else 0
        bucket_f1.append(f1); bucket_n.append(len(gt_b))

    print("\n" + "=" * 55)
    print("SECTION 9 — Length Performance Curve")
    print("=" * 55)
    print(f"  {'Word count':>12}  {'N':>6}  {'F1':>8}")
    for lbl, f, n in zip(labels, bucket_f1, bucket_n):
        print(f"  {lbl:>12}  {n:>6}  {f:>8.4f}")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax1.plot(labels, bucket_f1, "o-", color="steelblue", lw=2, label="F1")
    ax2.bar(labels, bucket_n, alpha=0.25, color="grey", label="Sample count")
    ax1.set_xlabel("Text length (word count bucket)")
    ax1.set_ylabel("F1 score", color="steelblue")
    ax2.set_ylabel("Sample count", color="grey")
    ax1.set_title("F1 Performance vs. Text Length")
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="grey")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    plt.tight_layout()
    path = OUT_DIR / "fig8_length_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  → {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    gt_pids, gt_labels = load_ground_truth()
    preds              = load_predictions()
    tsv                = load_tsv()

    assert len(gt_labels) == len(preds), \
        f"Length mismatch: {len(gt_labels)} GT vs {len(preds)} predictions"
    print(f"  {len(gt_labels)} dev samples | {sum(gt_labels)} PCL positive | "
          f"TSV overlap: {len(set(gt_pids) & set(tsv))}\n")

    section1_overall(gt_labels, preds)
    section2_category(gt_pids, gt_labels, preds, tsv)
    section3_annotator(gt_pids, gt_labels, preds, tsv)
    section4_examples(gt_pids, gt_labels, preds, tsv)
    section5_model_comparison()

    # Sections 6-8 require raw probabilities
    probs = load_probs()
    if probs is not None:
        section6_pr_curve(gt_labels, probs)
        section7_threshold_curve(gt_labels, probs)
        section8_calibration(gt_labels, probs)

    section9_length_curve(gt_pids, gt_labels, preds, tsv)

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
