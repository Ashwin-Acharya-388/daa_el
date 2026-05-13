"""
config.py — Centralized Configuration for the Financial Risk Analysis Tool.

All tuneable hyperparameters, file paths, and settings live here.
Modify this file to adapt the tool to different datasets or experiments.
"""

import os

# ─────────────────────────────── Paths ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Financial Distress.csv")
TAIWANESE_DATA_PATH = os.path.join(BASE_DIR, "taiwanese_bankruptcy.csv")
CONSOLIDATED_DATA_PATH = os.path.join(BASE_DIR, "consolidated_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Create output directories
for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────── Dataset Settings ────────────────────────
TARGET_COLUMN = "Financial Distress"       # Name of the target column in CSV
COLUMNS_TO_DROP = ["Company", "Time"]      # Non-feature columns to drop
BINARIZE_TARGET = True                     # Whether to convert target to 0/1
BINARIZE_THRESHOLD = -0.50                 # Values < threshold → 1 (distressed)
TEST_SIZE = 0.20                           # Fraction of data for test set
RANDOM_SEED = 42                           # Global random seed for reproducibility

# ─────────────────────────── DenseNet Architecture ───────────────────
GROWTH_RATE = 32          # Number of new features added per dense layer
NUM_DENSE_BLOCKS = 3      # Number of dense blocks
LAYERS_PER_BLOCK = 4      # Dense layers within each block
BOTTLENECK_FACTOR = 4     # Bottleneck dense layer width = BOTTLENECK_FACTOR * GROWTH_RATE
COMPRESSION = 0.5         # Transition layer compression factor
DROPOUT_RATE = 0.3        # Dropout in transition layers and classifier head

# ─────────────────────────── Training Settings ───────────────────────
EPOCHS = 200              # Maximum training epochs (early stopping will cut short)
MIN_EPOCHS = 20           # MINIMUM epochs to train (early stopping disabled before this)
BATCH_SIZE = 32           # Mini-batch size
LEARNING_RATE = 0.001     # Initial learning rate for Adam optimizer
EARLY_STOP_PATIENCE = 25  # Stop if val_loss doesn't improve for N epochs
LR_REDUCE_PATIENCE = 10   # Reduce LR if val_loss doesn't improve for N epochs
LR_REDUCE_FACTOR = 0.5    # Multiply LR by this factor on plateau
MIN_LR = 1e-6             # Floor for learning rate reduction

# ─────────────────────────── SMOTE-ENN Settings ──────────────────────
SMOTE_SAMPLING_STRATEGY = "auto"   # "auto" = balance to equal classes
SMOTE_K_NEIGHBORS = 5             # k-neighbors for SMOTE synthetic generation
ENN_N_NEIGHBORS = 3               # k-neighbors for ENN cleaning

# ─────────────────────────── SHAP Settings ───────────────────────────
SHAP_BACKGROUND_SAMPLES = 100     # Samples used as SHAP background (k-means)
SHAP_NUM_EXPLANATIONS = 5         # Number of individual predictions to explain
SHAP_TOP_FEATURES = 10            # Top N features to show in explanations

# ─────────────────────── Greedy Feature Selection ────────────────────
GREEDY_MAX_FEATURES = 30          # Maximum features to select
GREEDY_MIN_IMPROVEMENT = 0.001    # Minimum AUC improvement (ε) to continue
GREEDY_CV_FOLDS = 5               # Cross-validation folds for evaluation
SELECTED_FEATURES_PATH = os.path.join(REPORTS_DIR, "selected_features.json")
USE_CONSOLIDATED = True           # Whether to use consolidated dataset
