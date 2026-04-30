"""
evaluation.py — Module E: Model Evaluation and Visualization.

Provides:
  - Full classification metrics (Accuracy, Precision, Recall, F1, AUC, MCC)
  - Confusion matrix heatmap
  - ROC curve with AUC
  - Training history curves (loss + accuracy over epochs)
  - Saves all metrics to JSON and plots to outputs/plots/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    matthews_corrcoef
)

import config


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: list = None) -> dict:
    """
    Comprehensive evaluation of the trained model.

    Parameters
    ----------
    model : keras.Model — trained model
    X_test : np.ndarray — test features (scaled)
    y_test : np.ndarray — test labels (0/1)
    feature_names : list — feature column names (optional)

    Returns
    -------
    metrics : dict — all computed metrics
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # ── Predictions ──
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    # ── Metrics ──
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "mcc": round(float(mcc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "test_samples": int(len(y_test)),
        "positive_samples": int(y_test.sum()),
        "negative_samples": int(len(y_test) - y_test.sum())
    }

    # ── Print Results ──
    print(f"\n  Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall (Sens.):    {rec:.4f}")
    print(f"  Specificity:       {specificity:.4f}")
    print(f"  F1-Score:          {f1:.4f}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  MCC:               {mcc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:4d}  FP={fp:4d}")
    print(f"    FN={fn:4d}  TP={tp:4d}")

    # ── Classification Report ──
    report = classification_report(y_test, y_pred,
                                   target_names=["Healthy (0)", "Distressed (1)"])
    print(f"\n{report}")

    # ── Save report ──
    report_path = os.path.join(config.REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DenseNet Financial Risk Analysis — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy:       {acc:.4f}\n")
        f.write(f"Precision:      {prec:.4f}\n")
        f.write(f"Recall:         {rec:.4f}\n")
        f.write(f"Specificity:    {specificity:.4f}\n")
        f.write(f"F1-Score:       {f1:.4f}\n")
        f.write(f"AUC-ROC:        {auc:.4f}\n")
        f.write(f"MCC:            {mcc:.4f}\n\n")
        f.write(report)
    print(f"[Eval] Classification report saved to: {report_path}")

    # ── Save metrics JSON ──
    json_path = os.path.join(config.REPORTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Eval] Metrics saved to: {json_path}")

    # ── Generate Plots ──
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_proba, auc)

    return metrics


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """Plot and save confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Healthy (0)", "Distressed (1)"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16}, linewidths=0.5,
                linecolor="gray", ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix — DenseNet", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix saved to: {path}")


def _plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, auc_score: float):
    """Plot and save ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2.5,
            label=f"DenseNet (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#e74c3c")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — DenseNet Financial Risk Model",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] ROC curve saved to: {path}")


def plot_training_history(history: dict):
    """
    Plot training/validation loss and accuracy curves.

    Parameters
    ----------
    history : dict — from model.fit().history
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    # ── Loss ──
    axes[0].plot(epochs, history["loss"], label="Train Loss",
                 color="#3498db", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss",
                 color="#e74c3c", lw=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy ──
    axes[1].plot(epochs, history["accuracy"], label="Train Accuracy",
                 color="#3498db", lw=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy",
                 color="#e74c3c", lw=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("DenseNet Training History", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Training history plot saved to: {path}")

    # Also plot AUC history if available
    if "auc" in history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history["auc"], label="Train AUC",
                color="#2ecc71", lw=2)
        ax.plot(epochs, history["val_auc"], label="Val AUC",
                color="#e67e22", lw=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_title("Training & Validation AUC-ROC", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        auc_path = os.path.join(config.PLOTS_DIR, "training_auc.png")
        plt.savefig(auc_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Eval] AUC history plot saved to: {auc_path}")


# ─────────────────────── Standalone Test ─────────────────────────────
if __name__ == "__main__":
    print("Evaluation module loaded. Use via train.py for full pipeline.")
