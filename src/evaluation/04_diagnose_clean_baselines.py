"""
04_diagnose_clean_baselines.py

Diagnostic analysis of clean baseline models for the
ML-NIDS adversarial robustness study.

Purpose:
    Investigate the gap between validation and test performance
    before beginning adversarial attack experiments.

This script does NOT:
    - retrain models
    - tune hyperparameters
    - modify the test set
    - perform adversarial attacks

It only performs post-hoc diagnostic analysis.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from scipy.stats import ks_2samp


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "results" / "models"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of features to report in the distribution analysis
TOP_FEATURES = 20


# ============================================================
# 3. LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("CLEAN BASELINE DIAGNOSTIC ANALYSIS")
print("=" * 70)

print("\nLoading processed data...")

X_train = pd.read_csv(
    DATA_PROCESSED_DIR / "unsw_X_train.csv"
)

X_val = pd.read_csv(
    DATA_PROCESSED_DIR / "unsw_X_val.csv"
)

X_test = pd.read_csv(
    DATA_PROCESSED_DIR / "unsw_X_test.csv"
)

y_train = pd.read_csv(
    DATA_PROCESSED_DIR / "unsw_y_train.csv"
).squeeze("columns")

y_val = pd.read_csv(
    DATA_PROCESSED_DIR / "unsw_y_val.csv"
).squeeze("columns")

y_test = pd.read_csv(
    DATA_PROCESSED_DIR / "unsw_y_test.csv"
).squeeze("columns")


print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")


# ============================================================
# 4. BASIC DATASET COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

def class_distribution(y):
    counts = y.value_counts().sort_index()
    proportions = y.value_counts(
        normalize=True
    ).sort_index()

    return {
        "class_0_count": int(counts.get(0, 0)),
        "class_1_count": int(counts.get(1, 0)),
        "class_0_proportion": float(proportions.get(0, 0)),
        "class_1_proportion": float(proportions.get(1, 0)),
    }


distribution_summary = {
    "train": class_distribution(y_train),
    "validation": class_distribution(y_val),
    "test": class_distribution(y_test),
}

for dataset_name, values in distribution_summary.items():

    print(f"\n{dataset_name.upper()}")

    for key, value in values.items():
        print(f"{key}: {value}")


# Save class distribution

with open(
    LOG_DIR / "baseline_class_distribution.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        distribution_summary,
        f,
        indent=4
    )


# ============================================================
# 5. LOAD CLEAN BASELINE MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING CLEAN BASELINE MODELS")
print("=" * 70)

models = {
    "MLP": MODEL_DIR / "mlp_clean.joblib",
    "RandomForest": MODEL_DIR / "randomforest_clean.joblib",
    "XGBoost": MODEL_DIR / "xgboost_clean.joblib",
}

loaded_models = {}

for model_name, model_path in models.items():

    print(f"\nLoading {model_name}...")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    loaded_models[model_name] = joblib.load(model_path)

    print("Loaded successfully.")


# ============================================================
# 6. PREPARE NUMPY DATA
# ============================================================

X_train_np = X_train.to_numpy(dtype=np.float32)
X_val_np = X_val.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)

y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()
y_test_np = y_test.to_numpy()


# ============================================================
# 7. MODEL PREDICTION DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PREDICTION DIAGNOSTICS")
print("=" * 70)

prediction_summary = []
all_test_predictions = {}


for model_name, model in loaded_models.items():

    print(f"\nAnalyzing {model_name}...")

    train_pred = model.predict(X_train_np)
    val_pred = model.predict(X_val_np)
    test_pred = model.predict(X_test_np)

    train_prob = model.predict_proba(X_train_np)[:, 1]
    val_prob = model.predict_proba(X_val_np)[:, 1]
    test_prob = model.predict_proba(X_test_np)[:, 1]

    all_test_predictions[model_name] = test_pred

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test_np,
        test_pred,
        labels=[0, 1]
    ).ravel()

    prediction_summary.append({
        "model": model_name,

        "train_accuracy": float(
            np.mean(train_pred == y_train_np)
        ),

        "validation_accuracy": float(
            np.mean(val_pred == y_val_np)
        ),

        "test_accuracy": float(
            np.mean(test_pred == y_test_np)
        ),

        "test_true_negatives": int(tn),
        "test_false_positives": int(fp),
        "test_false_negatives": int(fn),
        "test_true_positives": int(tp),

        "test_mean_positive_probability": float(
            np.mean(test_prob)
        ),

        "test_median_positive_probability": float(
            np.median(test_prob)
        ),

        "test_std_positive_probability": float(
            np.std(test_prob)
        ),

        "test_min_positive_probability": float(
            np.min(test_prob)
        ),

        "test_max_positive_probability": float(
            np.max(test_prob)
        ),

        "test_roc_auc": float(
            roc_auc_score(y_test_np, test_prob)
        ),

        "test_pr_auc": float(
            average_precision_score(
                y_test_np,
                test_prob
            )
        ),
    })


prediction_summary_df = pd.DataFrame(
    prediction_summary
)

prediction_summary_df.to_csv(
    TABLE_DIR / "baseline_prediction_diagnostics.csv",
    index=False
)


# ============================================================
# 8. TRAINING VS TEST ACCURACY GAP
# ============================================================

print("\n" + "=" * 70)
print("TRAINING / VALIDATION / TEST PERFORMANCE GAP")
print("=" * 70)

gap_rows = []

for model_name, model in loaded_models.items():

    train_pred = model.predict(X_train_np)
    val_pred = model.predict(X_val_np)
    test_pred = model.predict(X_test_np)

    train_accuracy = np.mean(
        train_pred == y_train_np
    )

    val_accuracy = np.mean(
        val_pred == y_val_np
    )

    test_accuracy = np.mean(
        test_pred == y_test_np
    )

    gap_rows.append({
        "model": model_name,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_to_test_gap": train_accuracy - test_accuracy,
        "validation_to_test_gap": val_accuracy - test_accuracy,
    })


gap_df = pd.DataFrame(gap_rows)

print(
    gap_df.to_string(index=False)
)

gap_df.to_csv(
    TABLE_DIR / "baseline_performance_gaps.csv",
    index=False
)


# ============================================================
# 9. FEATURE DISTRIBUTION SHIFT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)

print(
    "\nComparing training and test feature distributions "
    "using the two-sample Kolmogorov-Smirnov test."
)

feature_shift_rows = []

for feature in X_train.columns:

    train_values = X_train[feature].to_numpy()
    test_values = X_test[feature].to_numpy()

    statistic, p_value = ks_2samp(
        train_values,
        test_values
    )

    feature_shift_rows.append({
        "feature": feature,
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
    })


feature_shift_df = pd.DataFrame(
    feature_shift_rows
)

feature_shift_df = feature_shift_df.sort_values(
    by="ks_statistic",
    ascending=False
)

feature_shift_df.to_csv(
    TABLE_DIR / "train_test_feature_distribution_shift.csv",
    index=False
)


print("\nTop features by KS statistic:")

print(
    feature_shift_df.head(TOP_FEATURES).to_string(
        index=False
    )
)


# ============================================================
# 10. MODEL DISAGREEMENT
# ============================================================

print("\n" + "=" * 70)
print("MODEL DISAGREEMENT")
print("=" * 70)

prediction_matrix = pd.DataFrame(
    all_test_predictions
)

prediction_matrix["y_true"] = y_test_np

prediction_matrix.to_csv(
    TABLE_DIR / "baseline_test_prediction_comparison.csv",
    index=False
)


# Number of models predicting each observation as attack

prediction_matrix["number_of_attack_predictions"] = (
    prediction_matrix[
        ["MLP", "RandomForest", "XGBoost"]
    ].sum(axis=1)
)

disagreement_counts = (
    prediction_matrix[
        "number_of_attack_predictions"
    ]
    .value_counts()
    .sort_index()
)

print("\nNumber of models predicting attack:")

print(disagreement_counts)


disagreement_summary = {
    str(int(index)): int(value)
    for index, value in disagreement_counts.items()
}

with open(
    LOG_DIR / "model_disagreement.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        disagreement_summary,
        f,
        indent=4
    )


# ============================================================
# 11. ERROR AGREEMENT
# ============================================================

print("\n" + "=" * 70)
print("ERROR AGREEMENT BETWEEN MODELS")
print("=" * 70)

error_matrix = pd.DataFrame(index=loaded_models.keys())

for model_name in loaded_models.keys():

    predictions = all_test_predictions[model_name]

    error_matrix.loc[
        model_name,
        "total_errors"
    ] = np.sum(predictions != y_test_np)

    error_matrix.loc[
        model_name,
        "false_positives"
    ] = np.sum(
        (predictions == 1) &
        (y_test_np == 0)
    )

    error_matrix.loc[
        model_name,
        "false_negatives"
    ] = np.sum(
        (predictions == 0) &
        (y_test_np == 1)
    )

print(error_matrix)


error_matrix.to_csv(
    TABLE_DIR / "baseline_error_summary.csv"
)


# ============================================================
# 12. CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

for model_name in loaded_models.keys():

    predictions = all_test_predictions[model_name]

    cm = confusion_matrix(
        y_test_np,
        predictions,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Normal", "Actual_Attack"],
        columns=["Predicted_Normal", "Predicted_Attack"]
    )

    print(f"\n{model_name}:")
    print(cm_df)

    cm_df.to_csv(
        TABLE_DIR / (
            f"{model_name.lower()}_baseline_confusion_matrix.csv"
        )
    )


# ============================================================
# 13. SAVE COMPLETE DIAGNOSTIC SUMMARY
# ============================================================

diagnostic_summary = {
    "experiment": "clean_baseline_diagnostics",

    "dataset": "UNSW-NB15",

    "random_state": RANDOM_STATE,

    "purpose": (
        "Post-hoc diagnostic analysis of clean baseline "
        "models before adversarial evaluation."
    ),

    "models": list(loaded_models.keys()),

    "training_samples": int(X_train.shape[0]),
    "validation_samples": int(X_val.shape[0]),
    "test_samples": int(X_test.shape[0]),

    "processed_features": int(X_train.shape[1]),

    "test_set_modified": False,

    "models_retrained": False,

    "hyperparameters_modified": False,

    "adversarial_attacks_performed": False,

    "feature_distribution_test": (
        "Two-sample Kolmogorov-Smirnov test"
    ),

    "outputs": {
        "class_distribution":
            "baseline_class_distribution.json",

        "prediction_diagnostics":
            "baseline_prediction_diagnostics.csv",

        "performance_gaps":
            "baseline_performance_gaps.csv",

        "feature_distribution_shift":
            "train_test_feature_distribution_shift.csv",

        "prediction_comparison":
            "baseline_test_prediction_comparison.csv",

        "model_disagreement":
            "model_disagreement.json",

        "error_summary":
            "baseline_error_summary.csv",
    },
}


with open(
    LOG_DIR / "clean_baseline_diagnostic_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        diagnostic_summary,
        f,
        indent=4
    )


# ============================================================
# 14. FINISH
# ============================================================

print("\n" + "=" * 70)
print("BASELINE DIAGNOSTIC ANALYSIS COMPLETED")
print("=" * 70)

print("\nDiagnostic tables saved to:")
print(TABLE_DIR)

print("\nDiagnostic logs saved to:")
print(LOG_DIR)

print("\nNo models were retrained.")
print("No hyperparameters were changed.")
print("No adversarial attacks were performed.")
print("The test set was not modified.")

print("\n" + "=" * 70)
print("NEXT STEP: REVIEW DIAGNOSTIC RESULTS")
print("=" * 70)