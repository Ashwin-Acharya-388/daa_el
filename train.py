"""
train.py — Main Training Orchestrator.

Runs the full pipeline:
  1. Data preprocessing (load, clean, scale, split)
  2. SMOTE-ENN balancing
  3. Build + compile DenseNet
  4. Train with callbacks
  5. Evaluate on test set
  6. Generate SHAP explanations
  7. Save all artifacts

Usage:
    python train.py
    python train.py --skip-shap    # Skip SHAP (faster, for debugging)
    python train.py --data my.csv  # Use custom dataset
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

import config

# ── Reproducibility ──
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(config.RANDOM_SEED)


def main():
    parser = argparse.ArgumentParser(
        description="DenseNet Financial Risk Analysis — Training Pipeline"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset CSV (default: Financial Distress.csv)")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP analysis (faster training)")
    parser.add_argument("--consolidated", action="store_true",
                        help="Use consolidated dataset (Financial Distress + Taiwanese)")
    parser.add_argument("--use-selected-features", action="store_true",
                        help="Use greedy-selected features from greedy_feature_selection.py")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  DenseNet Financial Risk Analysis Tool")
    print("  Loan Default Prediction with SHAP Explainability")
    print("=" * 70)
    print(f"\n  TensorFlow version: {tf.__version__}")
    print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"  Random seed: {config.RANDOM_SEED}")

    # ════════════════════════════════════════════════════════════════════
    #  STEP 1: Data Preprocessing
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*60}")
    print("STEP 1: DATA PREPROCESSING")
    print(f"{'━'*60}")

    if args.consolidated or config.USE_CONSOLIDATED:
        # Use consolidated dataset (merged Financial Distress + Taiwanese)
        print("[INFO] Using CONSOLIDATED dataset (Financial Distress + Taiwanese)")
        from data_consolidation import consolidate_datasets
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import json

        consolidated = consolidate_datasets()
        X_all = consolidated["X"]
        y_all = consolidated["y"]
        feature_names = consolidated["feature_names"]

        # Apply greedy-selected features if requested
        if args.use_selected_features and os.path.exists(config.SELECTED_FEATURES_PATH):
            with open(config.SELECTED_FEATURES_PATH, "r") as f:
                sel = json.load(f)
            selected_features = sel["selected_features"]
            print(f"[INFO] Using {len(selected_features)} greedy-selected features")
            X_all = X_all[selected_features]
            feature_names = selected_features

        # Stratified split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all, y_all.values,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=y_all.values
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # Save scaler
        import joblib
        scaler_path = os.path.join(config.MODEL_DIR, "scaler.joblib")
        joblib.dump(scaler, scaler_path)

        print(f"  Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        print(f"  Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")

    else:
        from data_preprocessing import run_preprocessing
        data = run_preprocessing(args.data)

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        feature_names = data["feature_names"]

    # ════════════════════════════════════════════════════════════════════
    #  STEP 2: Data Balancing (SMOTE-ENN)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*60}")
    print("STEP 2: SMOTE-ENN DATA BALANCING")
    print(f"{'━'*60}")

    from data_balancing import apply_smote_enn
    X_train_bal, y_train_bal = apply_smote_enn(X_train, y_train)

    # ════════════════════════════════════════════════════════════════════
    #  STEP 3: Build DenseNet Model
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*60}")
    print("STEP 3: BUILDING DENSENET MODEL")
    print(f"{'━'*60}")

    from densenet_model import (
        build_densenet, compile_model, compute_weights, train_model
    )

    n_features = X_train_bal.shape[1]
    model = build_densenet(n_features)
    model = compile_model(model)

    # Print architecture summary
    model.summary()
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")

    # Compute class weights (even after SMOTE-ENN, slight imbalance may remain)
    class_weight = compute_weights(y_train_bal)

    # ════════════════════════════════════════════════════════════════════
    #  STEP 4: Train Model
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*60}")
    print("STEP 4: TRAINING DENSENET")
    print(f"{'━'*60}")

    history = train_model(
        model, X_train_bal, y_train_bal,
        X_test, y_test,
        class_weight=class_weight
    )

    # ════════════════════════════════════════════════════════════════════
    #  STEP 5: Evaluate Model
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'━'*60}")
    print("STEP 5: MODEL EVALUATION")
    print(f"{'━'*60}")

    from evaluation import evaluate_model, plot_training_history

    metrics = evaluate_model(model, X_test, y_test, feature_names)
    plot_training_history(history)

    # ════════════════════════════════════════════════════════════════════
    #  STEP 6: SHAP Explanations
    # ════════════════════════════════════════════════════════════════════
    if not args.skip_shap:
        print(f"\n{'━'*60}")
        print("STEP 6: SHAP EXPLAINABILITY ANALYSIS")
        print(f"{'━'*60}")

        from shap_explainer import (
            create_explainer, compute_shap_values,
            plot_global_importance, generate_explanations
        )

        # Create explainer with background from training data
        explainer = create_explainer(model, X_train_bal, feature_names)

        # Compute SHAP values for a subset of test set
        n_explain = min(50, len(X_test))
        shap_values = compute_shap_values(
            explainer, X_test[:n_explain], feature_names
        )

        # Global importance plots
        plot_global_importance(shap_values, X_test[:n_explain], feature_names)

        # Individual explanations
        generate_explanations(model, explainer, X_test, feature_names)
    else:
        print(f"\n[INFO] SHAP analysis skipped (--skip-shap flag)")

    # ════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print("  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:   {elapsed/60:.1f} minutes")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  MCC:          {metrics['mcc']:.4f}")
    print(f"\n  Saved artifacts:")
    print(f"    Model:   {config.MODEL_DIR}/densenet_model.h5")
    print(f"    Scaler:  {config.MODEL_DIR}/scaler.joblib")
    print(f"    Plots:   {config.PLOTS_DIR}/")
    print(f"    Reports: {config.REPORTS_DIR}/")
    print(f"\n  To predict on new data:")
    print(f"    python prediction.py --single          # Single application")
    print(f"    python prediction.py --batch data.csv   # Batch prediction")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
