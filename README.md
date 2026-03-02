# SemEval 2022 Task 4 — Patronizing and Condescending Language Detection

**Task:** Binary classification — given a paragraph, predict whether it contains Patronizing and Condescending Language (PCL) towards a vulnerable group.
**Dataset:** DontPatronizeMe (Pérez-Almendros et al., 2020)
**Best model:** Probability-averaging ensemble of two fine-tuned RoBERTa-base classifiers.

---

## Quickstart: Finding the Predictions

The submission files are committed directly to the repo — no re-running is needed to mark them.

```
predictions/
├── dev.txt    # 2094 lines, one prediction per line (0 = No PCL, 1 = PCL)
└── test.txt   # 3832 lines, one prediction per line
```

`dev.txt` was generated using a model trained on the training set only (no dev leakage).
`test.txt` was generated using a model trained on train + dev combined.

---

## Repository Structure

```
.
├── predictions/              # Final submission files (dev.txt, test.txt)
├── BestModel/
│   └── best_model.ipynb      # Walkthrough notebook — start here to reproduce results
├── stage2/
│   ├── stage2.py             # Exploratory Data Analysis (EDA)
│   ├── eda_1_binary.png      # Class distribution plot
│   └── eda_2_train_ngrams.png# N-gram frequency analysis
├── stage4/
│   ├── stage4_final.py       # Full training pipeline (HPO → compare → retrain → predict)
│   └── outputs_stage4/       # Saved checkpoints and results
│       ├── comparison_results.json   # Dev F1 for each approach
│       ├── final_config.json         # Thresholds and checkpoint paths used for predictions
│       └── hpo_roberta_*_checkpoint/ # Best HPO checkpoints (used for dev.txt)
├── stage5/
│   ├── error_analysis.py     # Generates all error analysis figures and tables
│   ├── save_probs.py         # Generates dev_probs.npy (needs GPU + checkpoints)
│   ├── dev_probs.npy         # Cached ensemble probabilities on dev set
│   ├── fig1_confusion_matrix.png
│   ├── fig2_category_miss_rate.png
│   ├── fig3_annotator_score.png
│   ├── fig4_model_comparison.png
│   ├── fig5_pr_curve.png
│   ├── fig6_threshold_curve.png
│   ├── fig7_calibration.png
│   ├── fig8_length_curve.png
│   ├── fp_cases.csv          # All false positive examples
│   └── fn_cases.csv          # All false negative examples
├── dataset/                  # Dataset files (not committed — place here before running)
│   └── train/
│       ├── data/dontpatronizeme_pcl.tsv
│       └── labels/
│           ├── train_semeval_parids-labels.csv
│           └── dev_semeval_parids-labels.csv
│   └── test/
│       └── data/task4_test.tsv
└── requirements.txt
```

---

## Approach Summary

The dataset has severe class imbalance (~9.5% positive). A single loss function creates a precision-recall trade-off:

| Model | Dev F1 | Precision | Recall |
|---|---|---|---|
| RoBERTa + cross-entropy | 0.5954 | 0.603 | 0.588 |
| RoBERTa + focal loss | 0.5948 | 0.557 | 0.638 |
| **Ensemble (average probs)** | **0.5967** | 0.568 | 0.628 |
| DeBERTa-v3-base | 0.173 | 0.095 | 1.000 |

Averaging the two RoBERTa models balances their complementary biases (CE is more precise; focal loss is more sensitive), achieving the best F1.
DeBERTa collapsed to predicting all positives due to batch-level class imbalance instability.

---

## Reproducing Results (requires GPU)

The notebook **`BestModel/best_model.ipynb`** walks through each step. Alternatively, run the pipeline directly:

### 1. Install dependencies

```bash
# GPU (Azure / CUDA 12.x) — install torch first:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Step 1 — Hyperparameter optimisation (5 trials per model, ~1–2 hrs on GPU)
python stage4/stage4_final.py --mode hpo

# Step 2 — Compare models on official dev set
python stage4/stage4_final.py --mode compare

# Step 3 — Retrain on full data
python stage4/stage4_final.py --mode retrain

# Step 4 — Generate predictions
python stage4/stage4_final.py --mode predict
# Outputs: stage4/outputs_stage4/dev.txt and test.txt
# (also committed to predictions/ for convenience)
```

Or run all steps at once:
```bash
python stage4/stage4_final.py --mode all
```

### 3. Reproduce EDA figures

```bash
python stage2/stage2.py
# Outputs: stage2/eda_1_binary.png, stage2/eda_2_train_ngrams.png
```

### 4. Reproduce error analysis figures

All figures in `stage5/` are pre-generated and committed. To regenerate:

```bash
python stage5/error_analysis.py
# Outputs: stage5/fig1–fig8 .png, fp_cases.csv, fn_cases.csv
```

> `dev_probs.npy` (ensemble probabilities on the dev set) is already committed.
> If it needs to be regenerated, run `python stage5/save_probs.py` on a machine with GPU and the HPO checkpoints.

---

## Dataset Setup

The dataset is not included in the repo. Place files as follows before running:

```
dataset/train/data/dontpatronizeme_pcl.tsv
dataset/train/labels/train_semeval_parids-labels.csv
dataset/train/labels/dev_semeval_parids-labels.csv
dataset/test/data/task4_test.tsv
```
