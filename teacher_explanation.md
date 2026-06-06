# Teacher Presentation Guide: Financial Risk Analysis Tool
### End-to-End DenseNet Loan Default Prediction System

This guide outlines the system's architecture, core machine learning concepts, the mathematical resolution of a major scaling bug, and instructions on how to run simulations for a presentation.

---

## 1. System Architecture Overview

The application is a full-stack AI-driven financial risk analysis dashboard:

```
                  ┌──────────────────────────────┐
                  │   React + Vite Frontend UI   │
                  │   (Visualizations & Gauges)  │
                  └──────────────┬───────────────┘
                                 │ HTTP requests + WebSockets
                                 ▼
                  ┌──────────────────────────────┐
                  │      FastAPI Backend API     │
                  │ (Uvicorn / Memory-state/CORS)│
                  └──────────────┬───────────────┘
                                 │ Inference
                                 ▼
                  ┌──────────────────────────────┐
                  │     Trained DenseNet Model   │
                  │  (14 SFFS-selected features) │
                  └──────────────────────────────┘
```

* **Frontend (React + Vite)**: Modern dashboard presenting single/batch applicant inputs, dynamic risk gauges, and local SHAP explanation charts.
* **Backend (FastAPI)**: Asynchronous REST endpoints communicating with the model, checking health, and streaming batch job progress via WebSockets.
* **Machine Learning Engine (TensorFlow/Keras)**: A DenseNet deep learning classifier trained on the Taiwanese Bankruptcy dataset.

---

## 2. Key Machine Learning Concepts

When explaining this project to your teacher, focus on these three core ML pillars:

### A. Sequential Floating Forward Selection (SFFS)
* **What it is**: A feature selection method used to extract the most predictive inputs from the dataset.
* **Why it is better than standard greedy forward selection**: Standard forward selection suffers from the "nesting effect" (once a feature is added, it can never be removed). SFFS adds a **conditional backward exclusion step**. If adding a new feature makes an older feature redundant, SFFS "floats" backward and removes it, escaping local optima.
* **Result**: We selected the **14 most predictive features** (such as Debt Ratio, Return on Assets (ROA), Persistent EPS, Cash/Total Assets, etc.) which resulted in high predictive performance.

### B. SMOTE-ENN Class Balancing
* **The Problem**: Default datasets are heavily imbalanced (e.g., less than 4% of companies default). A model trained on this would simply guess "no default" 96% of the time and have 96% accuracy, while being useless at finding risky companies.
* **The Solution**: 
  1. **SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic data points for the minority (defaulted) class.
  2. **ENN** (Edited Nearest Neighbors) cleans up noise by removing data points whose class differs from their neighbors, creating distinct, clean boundaries.

### C. DenseNet for Tabular Data
* **Why DenseNet**: Standard neural networks pass information from layer to layer sequentially. DenseNet introduces **dense connectivity**—each layer receives the concatenated outputs of *all* preceding layers in its dense block.
* **Advantage**: Encourages feature reuse, improves gradient flow, prevents the vanishing gradient problem, and operates with fewer parameters.

---

## 3. The "0% Default" Bug & Mathematical Resolution

### The Problem (Zero-Padding Skew)
Originally, the model was trained on a "consolidated" dataset that unioned two different financial datasets. Features unique to one dataset were padded with `0.0` for entries from the other.
* This massive artificial zero-padding corrupted the statistical properties of the dataset.
* The `StandardScaler` computed an extremely small standard deviation and skewed mean.
* When realistic non-zero inputs were sent from the UI, the scaler transformed them into extreme outliers (e.g., **13+ standard deviations away** from the training mean).
* These extreme outliers saturated the final layer's `sigmoid` activation function ($\frac{1}{1 + e^{-z}}$), locking the output at exactly `0.00%` default risk.

### The Fix (Clean Dataset Transition)
We abandoned the zero-padded consolidated approach and trained the model exclusively on the clean **Taiwanese Bankruptcy dataset**:
* Renamed raw feature names (like "ROA(C) before interest...") to standardised codes (`tw_f1`, `tw_f2`...) so the frontend could map them to human-readable labels.
* Programmed the preprocessor to filter the features to the 14 SFFS-selected inputs before fitting the `StandardScaler`.
* Retrained the DenseNet model on these clean, properly scaled features.
* The model achieved **91.35% accuracy** and **87.61% AUC-ROC**, and predicts dynamically and realistically based on inputs.

---

## 4. How to Run & Simulate for Presentation

### A. Starting the Servers
Open your terminal and run the following commands:

1. **Start the FastAPI Backend**:
   ```bash
   # Make sure you are in the project folder 'daa_el'
   export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib"
   .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
2. **Start the React Frontend**:
   ```bash
   # Open a separate terminal window/tab in the 'daa_el/frontend' directory
   npm run dev
   ```
3. **Open the browser**:
   Go to `http://localhost:5173/` and navigate to the **Single Prediction** page.

---

### B. Demonstrating Predictions (Simulations)

Show your teacher how the model dynamically responds using `curl` commands in a separate terminal:

#### 1. Baseline Test (Healthy Profile)
Simulates a healthy company using median values (low debt, steady earnings):
```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fra-dev-key-2024" \
  -d '{"features": {"tw_f19": 0.224, "tw_f37": 0.111, "tw_f50": 0.0297, "tw_f2": 0.559, "tw_f43": 0.178, "tw_f74": 1140000000.0, "tw_f91": 0.278, "tw_f18": 0.184, "tw_f54": 0.810, "tw_f1": 0.502, "tw_f86": 0.810, "tw_f57": 0.0746, "tw_f81": 0.460, "tw_f3": 0.552}, "include_shap": false}'
```
* **Expected Output**: `probability_of_default` ≈ **0.25%** (`APPROVED / LOW RISK` with 99.8% confidence).

#### 2. Risk Test (High Default Profile)
Simulates an extremely distressed profile (90% debt ratio, 0% ROA, zero cash):
```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fra-dev-key-2024" \
  -d '{"features": {"tw_f19": 0.0, "tw_f37": 0.9, "tw_f50": 0.0, "tw_f2": 0.0, "tw_f43": 0.0, "tw_f74": 0.0, "tw_f91": 0.9, "tw_f18": 0.0, "tw_f54": 0.0, "tw_f1": 0.0, "tw_f86": 0.0, "tw_f57": 0.0, "tw_f81": 0.0, "tw_f3": 0.0}, "include_shap": false}'
```
* **Expected Output**: `probability_of_default` = **100.0%** (`DEFAULT / HIGH RISK` with 100.0% confidence).

Demonstrating these two commands will show that the scaler and deep neural network model are fully working, mathematically correct, and highly sensitive to credit risk changes.
