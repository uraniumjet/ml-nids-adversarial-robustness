"""
03_train_clean_baselines.py

Train and evaluate clean baseline ML models for the
ML-NIDS adversarial robustness study.

Datasets:
    UNSW-NB15

Models:
    1. MLP
    2. Random Forest
    3. XGBoost

Important:
    - The test set is used only for final evaluation.
    - No adversarial attacks are performed in this script.
    - Preprocessing was already fitted on the training data.
    - Results are saved for later comparison with adversarial experiments.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
RESULT_DIR = PROJECT_ROOT / "results" / "tables"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 3. LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("ML-NIDS CLEAN BASELINE TRAINING")
print("=" * 70)

print("\nLoading processed UNSW-NB15 data...")

X_train = pd.read_csv(DATA_DIR / "unsw_X_train.csv")
X_val = pd.read_csv(DATA_DIR / "unsw_X_val.csv")
X_test = pd.read_csv(DATA_DIR / "unsw_X_test.csv")

y_train = pd.read_csv(DATA_DIR / "unsw_y_train.csv").squeeze("columns")
y_val = pd.read_csv(DATA_DIR / "unsw_y_val.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "unsw_y_test.csv").squeeze("columns")


print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")

print(f"\ny_train distribution:")
print(y_train.value_counts().sort_index())

print(f"\ny_val distribution:")
print(y_val.value_counts().sort_index())

print(f"\ny_test distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# 4. CONVERT TO NUMPY
# ============================================================

X_train_np = X_train.to_numpy(dtype=np.float32)
X_val_np = X_val.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)

y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()
y_test_np = y_test.to_numpy()


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

models = {

    "LogisticRegression" : LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=True,
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model, X, y):
    """
    Evaluate a trained binary classifier.

    Returns:
        Dictionary containing performance metrics.
    """

    predictions = model.predict(X)

    # Probability of positive class
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = None

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    if probabilities is not None:
        roc_auc = roc_auc_score(y, probabilities)
        pr_auc = average_precision_score(y, probabilities)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "fnr": fnr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return results, predictions, probabilities


# ============================================================
# 7. TRAINING AND EVALUATION
# ============================================================

all_results = []

training_log = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train_np,
        y_train_np
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds.")

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    print("\nValidation evaluation...")

    val_results, _, _ = evaluate_model(
        model_name,
        model,
        X_val_np,
        y_val_np
    )

    print("\nValidation results:")
    for key, value in val_results.items():
        if key != "model":
            print(f"{key}: {value}")

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    print("\nFinal test evaluation...")

    test_results, test_predictions, test_probabilities = evaluate_model(
        model_name,
        model,
        X_test_np,
        y_test_np
    )

    test_results["training_time_seconds"] = training_time

    all_results.append(test_results)

    print("\nTest results:")
    for key, value in test_results.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = MODEL_DIR / f"{model_name.lower()}_clean.joblib"

    joblib.dump(
        model,
        model_path
    )

    print(f"\nModel saved to:")
    print(model_path)

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "y_true": y_test_np,
        "y_pred": test_predictions,
        "confidence": test_probabilities,
    })

    prediction_path = RESULT_DIR / f"{model_name.lower()}_clean_predictions.csv"

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(f"Predictions saved to:")
    print(prediction_path)

    # --------------------------------------------------------
    # Save validation results
    # --------------------------------------------------------

    validation_path = RESULT_DIR / f"{model_name.lower()}_clean_validation.json"

    with open(validation_path, "w", encoding="utf-8") as f:
        json.dump(
            val_results,
            f,
            indent=4
        )

    training_log.append({
        "model": model_name,
        "training_time_seconds": training_time,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
    })


# ============================================================
# 8. SAVE COMPARATIVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_path = RESULT_DIR / "clean_baseline_results.csv"

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 9. SAVE EXPERIMENT METADATA
# ============================================================

metadata = {
    "experiment": "clean_baseline",
    "dataset": "UNSW-NB15",
    "random_state": RANDOM_STATE,

    "data": {
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "processed_features": int(X_train.shape[1]),
    },

    "models": list(models.keys()),

    "test_set_used_for_final_evaluation_only": True,

    "adversarial_attacks_used": False,

    "training_log": training_log,
}


metadata_path = LOG_DIR / "clean_baseline_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN BASELINE EXPERIMENT COMPLETED")
print("=" * 70)

print("\nFinal test-set comparison:")
print()

display_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)

print("\nResults saved to:")
print(results_path)

print("\nMetadata saved to:")
print(metadata_path)

print("\nBaseline models saved to:")
print(MODEL_DIR)

print("\n" + "=" * 70)
print("NEXT STEP: VERIFY BASELINE RESULTS BEFORE ADVERSARIAL ATTACKS")
print("=" * 70)