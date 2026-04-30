# Financial Risk Analysis Tool — DenseNet Loan Default Prediction

A production-ready financial risk analysis system using **DenseNet** deep learning architecture for loan default / financial distress prediction, with **SHAP** explainability.

Based on: **Sayed et al. (2024)** — *"Machine Learning and Deep Learning for Loan Prediction in Banking"*, IEEE Access.

---

## Features

- **DenseNet Architecture** adapted for tabular data with dense connectivity, bottleneck layers, and transition layers
- **SMOTE-ENN** data balancing for handling imbalanced loan datasets
- **SHAP Explainability** — global feature importance + per-prediction explanations
- **Full Evaluation Suite** — Accuracy, Precision, Recall, F1, AUC-ROC, MCC, confusion matrix, ROC curve
- **Prediction Pipeline** — single application and batch CSV inference
- **Dataset-agnostic** — works with any CSV containing features + binary target

---

## Project Structure

```
├── config.py                  # All configurable parameters
├── data_preprocessing.py      # Data loading, cleaning, scaling, splitting
├── data_balancing.py          # SMOTE-ENN class balancing
├── densenet_model.py          # DenseNet architecture & training
├── shap_explainer.py          # SHAP global/local explanations
├── evaluation.py              # Metrics, confusion matrix, ROC curve
├── prediction.py              # Single & batch inference
├── train.py                   # Main orchestrator
├── requirements.txt           # Dependencies
├── Financial Distress.csv     # Dataset
└── outputs/
    ├── model/                 # Saved model, scaler, metadata
    ├── plots/                 # Visualizations
    └── reports/               # Metrics JSON, SHAP explanations
```

---

## Quick Start

### 1. Set Up Virtual Environment

```bash
cd "/Users/ashwinacharya/Downloads/daa financial risk analysis EL"

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Full training with SHAP (takes ~10-20 minutes)
python train.py

# Quick training without SHAP (takes ~2-5 minutes)
python train.py --skip-shap
```

### 3. Make Predictions

```bash
# Single application (demo with random values)
python prediction.py --single

# Batch prediction from CSV
python prediction.py --batch your_data.csv

# Fast prediction without SHAP explanation
python prediction.py --single --no-shap
```

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BINARIZE_THRESHOLD` | -0.50 | Financial Distress < threshold → distressed (1) |
| `TEST_SIZE` | 0.20 | Train/test split ratio |
| `GROWTH_RATE` | 32 | DenseNet growth rate |
| `NUM_DENSE_BLOCKS` | 3 | Number of dense blocks |
| `LAYERS_PER_BLOCK` | 4 | Layers per dense block |
| `COMPRESSION` | 0.5 | Transition layer compression |
| `DROPOUT_RATE` | 0.3 | Dropout rate |
| `EPOCHS` | 200 | Max epochs (early stopping active) |
| `BATCH_SIZE` | 32 | Training batch size |

---

## Using with a Different Dataset

1. Place your CSV in the project directory
2. Edit `config.py`:
   ```python
   DATA_PATH = os.path.join(BASE_DIR, "your_dataset.csv")
   TARGET_COLUMN = "your_target_column"
   COLUMNS_TO_DROP = ["id_column", "date_column"]
   BINARIZE_TARGET = False  # if target is already 0/1
   ```
3. Run `python train.py`

---

## Expected Outputs

After training, the `outputs/` directory will contain:

### Model Files (`outputs/model/`)
- `densenet_model.h5` — Trained model weights
- `densenet_best.h5` — Best model checkpoint
- `scaler.joblib` — Fitted StandardScaler
- `metadata.json` — Feature names, training medians

### Plots (`outputs/plots/`)
- `class_distribution.png` — Before/after SMOTE-ENN balancing
- `training_history.png` — Loss & accuracy curves
- `training_auc.png` — AUC-ROC over epochs
- `confusion_matrix.png` — Test set confusion matrix
- `roc_curve.png` — ROC curve with AUC score
- `shap_feature_importance.png` — Top features by SHAP
- `shap_summary_beeswarm.png` — SHAP beeswarm plot
- `shap_waterfall_sample_*.png` — Individual prediction explanations

### Reports (`outputs/reports/`)
- `metrics.json` — All evaluation metrics
- `classification_report.txt` — Full sklearn classification report
- `shap_explanations.json` — Structured SHAP explanations

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| Out of memory | Reduce `BATCH_SIZE` in `config.py` (try 16) |
| SHAP too slow | Use `--skip-shap` flag, or reduce `SHAP_BACKGROUND_SAMPLES` |
| Low accuracy | Try increasing `EPOCHS`, `LAYERS_PER_BLOCK`, or `GROWTH_RATE` |
| TensorFlow GPU issues | Set `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` in `train.py` |
| "Target column not found" | Check `TARGET_COLUMN` in `config.py` matches your CSV |
