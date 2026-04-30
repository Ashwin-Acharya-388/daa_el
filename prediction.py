"""
prediction.py — Module F: Single & Batch Prediction Pipeline.

Provides:
  - predict_single(features_dict) — predict for one loan application
  - predict_batch(csv_path)       — predict for a CSV of applications
  - Loads saved model, scaler, and column metadata
  - Returns probability of default + SHAP explanation
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

import config


def load_inference_artifacts():
    """
    Load saved model, scaler, and metadata for inference.

    Returns
    -------
    model, scaler, metadata : tuple
    """
    from tensorflow import keras

    model_path = os.path.join(config.MODEL_DIR, "densenet_model.h5")
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.joblib")
    meta_path = os.path.join(config.MODEL_DIR, "metadata.json")

    for path, name in [(model_path, "Model"), (scaler_path, "Scaler"),
                        (meta_path, "Metadata")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found at {path}. Run train.py first."
            )

    model = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    print(f"[Predict] Loaded model, scaler, and metadata from {config.MODEL_DIR}")
    return model, scaler, metadata


def predict_single(features: dict, explain: bool = True) -> dict:
    """
    Predict loan default risk for a single application.

    Parameters
    ----------
    features : dict
        Feature name → value mapping. Missing features are filled
        with training medians.
    explain : bool
        Whether to include SHAP explanation (slower but informative).

    Returns
    -------
    result : dict with keys:
        - probability_of_default : float
        - decision : str ("DEFAULT" or "APPROVED")
        - confidence : float (%)
        - explanation : dict (if explain=True)
    """
    model, scaler, metadata = load_inference_artifacts()
    feature_names = metadata["feature_names"]
    medians = metadata["training_medians"]

    # Build feature vector, filling missing with training medians
    row = {}
    missing = []
    for col in feature_names:
        if col in features:
            row[col] = float(features[col])
        else:
            row[col] = medians.get(col, 0.0)
            missing.append(col)

    if missing:
        print(f"[Predict] Filled {len(missing)} missing features with "
              f"training medians: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # Create DataFrame and scale
    df = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(df.values)

    # Predict
    proba = float(model.predict(X_scaled, verbose=0).flatten()[0])
    decision = "DEFAULT / HIGH RISK" if proba >= 0.5 else "APPROVED / LOW RISK"
    confidence = proba if proba >= 0.5 else (1 - proba)

    result = {
        "probability_of_default": round(proba, 4),
        "decision": decision,
        "confidence": round(confidence * 100, 1)
    }

    # SHAP explanation
    if explain:
        try:
            from shap_explainer import create_explainer, explain_single_prediction

            # Use a small background for speed
            bg = np.zeros((1, len(feature_names)))
            explainer = create_explainer(model, X_scaled, feature_names)
            exp = explain_single_prediction(
                explainer, X_scaled[0], proba, feature_names,
                instance_idx=999
            )
            result["explanation"] = exp
        except Exception as e:
            result["explanation"] = {"error": str(e)}
            print(f"[Predict] SHAP explanation failed: {e}")

    # Print result
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULT")
    print(f"{'='*50}")
    print(f"  Default Probability: {proba:.2%}")
    print(f"  Decision:            {decision}")
    print(f"  Confidence:          {confidence*100:.1f}%")

    return result


def predict_batch(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Predict loan default risk for a batch of applications from a CSV.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with feature columns.
    output_path : str, optional
        Path to save predictions CSV. Defaults to
        outputs/reports/batch_predictions.csv

    Returns
    -------
    pd.DataFrame with original data + prediction columns
    """
    model, scaler, metadata = load_inference_artifacts()
    feature_names = metadata["feature_names"]
    medians = metadata["training_medians"]

    # Load input
    df = pd.read_csv(csv_path)
    print(f"[Predict] Loaded batch: {df.shape[0]} applications from {csv_path}")

    # Prepare features
    X = pd.DataFrame()
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col].astype(float)
        else:
            X[col] = medians.get(col, 0.0)

    # Impute remaining NaN with medians
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(medians.get(col, 0.0), inplace=True)

    # Scale and predict
    X_scaled = scaler.transform(X.values)
    probas = model.predict(X_scaled, verbose=0).flatten()

    # Attach results
    df["default_probability"] = np.round(probas, 4)
    df["predicted_class"] = (probas >= 0.5).astype(int)
    df["decision"] = np.where(probas >= 0.5,
                              "DEFAULT / HIGH RISK",
                              "APPROVED / LOW RISK")

    # Save
    output_path = output_path or os.path.join(config.REPORTS_DIR,
                                               "batch_predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"[Predict] Batch predictions saved to: {output_path}")

    # Summary
    n_default = (probas >= 0.5).sum()
    n_approve = (probas < 0.5).sum()
    print(f"\n  Results Summary:")
    print(f"    Total applications: {len(probas)}")
    print(f"    Approved (low risk): {n_approve} ({n_approve/len(probas)*100:.1f}%)")
    print(f"    Default (high risk): {n_default} ({n_default/len(probas)*100:.1f}%)")
    print(f"    Average default probability: {probas.mean():.4f}")

    return df


# ─────────────────────── CLI Interface ───────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Financial Risk Prediction — Inference"
    )
    parser.add_argument("--single", action="store_true",
                        help="Run a single sample prediction (demo)")
    parser.add_argument("--batch", type=str, default=None,
                        help="Path to CSV file for batch prediction")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP explanation for faster prediction")
    args = parser.parse_args()

    if args.single:
        # Demo: predict with some dummy values
        sample = {f"x{i}": np.random.randn() for i in range(1, 84)}
        result = predict_single(sample, explain=not args.no_shap)
        print(json.dumps(result, indent=2, default=str))

    elif args.batch:
        results = predict_batch(args.batch)
        print(results[["default_probability", "predicted_class", "decision"]].head(10))

    else:
        print("Usage:")
        print("  python prediction.py --single          # Demo single prediction")
        print("  python prediction.py --batch data.csv   # Batch prediction")
        print("  python prediction.py --single --no-shap # Without SHAP (faster)")
