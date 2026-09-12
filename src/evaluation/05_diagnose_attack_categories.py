"""
05_diagnose_attack_categories.py

Analyze the original UNSW-NB15 training and testing partitions
to understand attack-category composition and model errors.

Purpose:
    Determine whether differences in clean test performance are
    associated with differences in attack-category composition.

Important:
    - No model is retrained.
    - No hyperparameters are changed.
    - No test observations are modified.
    - attack_cat is used only for post-hoc analysis.
    - attack_cat is NOT a model input.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "results" / "models"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FILE PATHS
# ============================================================

TRAINING_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TESTING_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


# ============================================================
# 3. LOAD ORIGINAL DATA
# ============================================================

print("=" * 70)
print("UNSW-NB15 ATTACK CATEGORY DIAGNOSTIC")
print("=" * 70)

print("\nLoading original UNSW-NB15 training data...")

train_raw = pd.read_csv(
    TRAINING_FILE
)

print(
    f"Training data shape: {train_raw.shape}"
)

print("\nLoading original UNSW-NB15 testing data...")

test_raw = pd.read_csv(
    TESTING_FILE
)

print(
    f"Testing data shape: {test_raw.shape}"
)


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = {
    "training": ["label", "attack_cat"],
    "testing": ["label", "attack_cat"],
}

for column in required_columns["training"]:

    if column not in train_raw.columns:
        raise ValueError(
            f"Required column '{column}' "
            f"not found in training data."
        )

for column in required_columns["testing"]:

    if column not in test_raw.columns:
        raise ValueError(
            f"Required column '{column}' "
            f"not found in testing data."
        )


# ============================================================
# 5. CLEAN ATTACK CATEGORY LABELS
# ============================================================

# Strip accidental whitespace without changing the
# semantic category values.

train_raw["attack_cat"] = (
    train_raw["attack_cat"]
    .astype(str)
    .str.strip()
)

test_raw["attack_cat"] = (
    test_raw["attack_cat"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 6. ATTACK CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ATTACK CATEGORY DISTRIBUTION")
print("=" * 70)


def category_distribution(df):

    counts = (
        df["attack_cat"]
        .value_counts()
        .sort_index()
    )

    proportions = (
        df["attack_cat"]
        .value_counts(normalize=True)
        .sort_index()
    )

    result = pd.DataFrame({
        "count": counts,
        "proportion": proportions,
    })

    return result


train_categories = category_distribution(
    train_raw
)

test_categories = category_distribution(
    test_raw
)


print("\nTRAINING PARTITION:")
print(train_categories)


print("\nTESTING PARTITION:")
print(test_categories)


# ============================================================
# 7. TRAIN/TEST CATEGORY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TRAINING VS TEST ATTACK CATEGORY COMPARISON")
print("=" * 70)


category_comparison = pd.DataFrame(
    index=sorted(
        set(train_categories.index)
        | set(test_categories.index)
    )
)

category_comparison["train_count"] = (
    train_categories["count"]
)

category_comparison["test_count"] = (
    test_categories["count"]
)

category_comparison["train_proportion"] = (
    train_categories["proportion"]
)

category_comparison["test_proportion"] = (
    test_categories["proportion"]
)

category_comparison = (
    category_comparison
    .fillna(0)
)

category_comparison["proportion_difference"] = (
    category_comparison["test_proportion"]
    - category_comparison["train_proportion"]
)


print(
    category_comparison.to_string()
)


category_comparison.to_csv(
    TABLE_DIR / "unsw_attack_category_train_test_comparison.csv"
)


# ============================================================
# 8. VERIFY BINARY LABEL / ATTACK CATEGORY CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("LABEL / ATTACK CATEGORY CONSISTENCY")
print("=" * 70)


print("\nTraining:")
print(
    pd.crosstab(
        train_raw["attack_cat"],
        train_raw["label"]
    )
)


print("\nTesting:")
print(
    pd.crosstab(
        test_raw["attack_cat"],
        test_raw["label"]
    )
)


train_cross = pd.crosstab(
    train_raw["attack_cat"],
    train_raw["label"]
)

test_cross = pd.crosstab(
    test_raw["attack_cat"],
    test_raw["label"]
)

train_cross.to_csv(
    TABLE_DIR / "unsw_training_attack_category_label_crosstab.csv"
)

test_cross.to_csv(
    TABLE_DIR / "unsw_testing_attack_category_label_crosstab.csv"
)


# ============================================================
# 9. LOAD CLEAN BASELINE MODELS
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
            f"Model not found: {model_path}"
        )

    loaded_models[model_name] = joblib.load(
        model_path
    )

    print("Loaded successfully.")


# ============================================================
# 10. LOAD PROCESSED TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING PROCESSED TEST DATA")
print("=" * 70)


X_test = pd.read_csv(
    PROCESSED_DIR / "unsw_X_test.csv"
)

y_test = pd.read_csv(
    PROCESSED_DIR / "unsw_y_test.csv"
).squeeze("columns")


print(
    f"Processed test shape: {X_test.shape}"
)

print(
    f"Test target shape: {y_test.shape}"
)


# ============================================================
# 11. VERIFY ROW ALIGNMENT
# ============================================================

print("\nChecking raw/processed test alignment...")

if len(test_raw) != len(X_test):

    raise ValueError(
        "Raw testing-set row count does not match "
        "processed test-set row count."
    )


if len(test_raw) != len(y_test):

    raise ValueError(
        "Raw testing-set row count does not match "
        "processed test target row count."
    )


# Verify labels are identical

raw_test_labels = (
    test_raw["label"]
    .to_numpy()
)

processed_test_labels = (
    y_test.to_numpy()
)

if not np.array_equal(
    raw_test_labels,
    processed_test_labels
):

    raise ValueError(
        "Raw and processed test labels are not identical. "
        "Row alignment must be investigated before continuing."
    )


print("Row counts match.")
print("Test labels match exactly.")


# ============================================================
# 12. GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING POST-HOC TEST PREDICTIONS")
print("=" * 70)


X_test_np = X_test.to_numpy(
    dtype=np.float32
)

y_test_np = y_test.to_numpy()


predictions = {}

for model_name, model in loaded_models.items():

    print(
        f"\nGenerating predictions for {model_name}..."
    )

    predictions[model_name] = model.predict(
        X_test_np
    )


# ============================================================
# 13. ATTACH PREDICTIONS FOR CATEGORY ANALYSIS
# ============================================================

analysis_df = pd.DataFrame({
    "label": y_test_np,
    "attack_cat": test_raw["attack_cat"].to_numpy(),
})


for model_name, model_predictions in predictions.items():

    analysis_df[
        f"{model_name}_prediction"
    ] = model_predictions


# ============================================================
# 14. FALSE NEGATIVES BY ATTACK CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("FALSE NEGATIVES BY ATTACK CATEGORY")
print("=" * 70)


fn_rows = []


for model_name in loaded_models.keys():

    prediction_column = (
        f"{model_name}_prediction"
    )

    attack_mask = (
        analysis_df["label"] == 1
    )

    fn_mask = (
        attack_mask
        & (analysis_df[prediction_column] == 0)
    )

    fn_data = analysis_df.loc[fn_mask]

    counts = (
        fn_data["attack_cat"]
        .value_counts()
    )

    total_attack_by_category = (
        analysis_df.loc[attack_mask]
        ["attack_cat"]
        .value_counts()
    )

    for category in total_attack_by_category.index:

        fn_count = int(
            counts.get(category, 0)
        )

        attack_count = int(
            total_attack_by_category.get(category, 0)
        )

        fn_rate = (
            fn_count / attack_count
            if attack_count > 0
            else 0.0
        )

        fn_rows.append({
            "model": model_name,
            "attack_category": category,
            "attack_count": attack_count,
            "false_negative_count": fn_count,
            "false_negative_rate": fn_rate,
        })


fn_df = pd.DataFrame(fn_rows)

fn_df = fn_df.sort_values(
    [
        "model",
        "false_negative_rate"
    ],
    ascending=[True, False]
)


print(
    fn_df.to_string(index=False)
)


fn_df.to_csv(
    TABLE_DIR / "baseline_false_negatives_by_attack_category.csv",
    index=False
)


# ============================================================
# 15. FALSE POSITIVES ON NORMAL TRAFFIC
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVES ON NORMAL TRAFFIC")
print("=" * 70)


fp_rows = []


for model_name in loaded_models.keys():

    prediction_column = (
        f"{model_name}_prediction"
    )

    normal_mask = (
        analysis_df["label"] == 0
    )

    fp_mask = (
        normal_mask
        & (analysis_df[prediction_column] == 1)
    )

    fp_count = int(
        fp_mask.sum()
    )

    normal_count = int(
        normal_mask.sum()
    )

    fpr = (
        fp_count / normal_count
        if normal_count > 0
        else 0.0
    )

    fp_rows.append({
        "model": model_name,
        "normal_samples": normal_count,
        "false_positive_count": fp_count,
        "false_positive_rate": fpr,
    })


fp_df = pd.DataFrame(fp_rows)

print(
    fp_df.to_string(index=False)
)

fp_df.to_csv(
    TABLE_DIR / "baseline_false_positives_summary.csv",
    index=False
)


# ============================================================
# 16. MODEL ERROR COMPARISON BY CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("MODEL ERROR COMPARISON BY ATTACK CATEGORY")
print("=" * 70)


category_error_rows = []


for category in sorted(
    analysis_df["attack_cat"].unique()
):

    category_mask = (
        analysis_df["attack_cat"] == category
    )

    category_count = int(
        category_mask.sum()
    )

    row = {
        "attack_category": category,
        "sample_count": category_count,
    }

    for model_name in loaded_models.keys():

        prediction_column = (
            f"{model_name}_prediction"
        )

        errors = (
            analysis_df.loc[
                category_mask,
                prediction_column
            ]
            !=
            analysis_df.loc[
                category_mask,
                "label"
            ]
        )

        error_count = int(
            errors.sum()
        )

        error_rate = (
            error_count / category_count
            if category_count > 0
            else 0.0
        )

        row[
            f"{model_name}_error_count"
        ] = error_count

        row[
            f"{model_name}_error_rate"
        ] = error_rate

    category_error_rows.append(row)


category_error_df = pd.DataFrame(
    category_error_rows
)


print(
    category_error_df.to_string(
        index=False
    )
)


category_error_df.to_csv(
    TABLE_DIR / "baseline_error_rate_by_attack_category.csv",
    index=False
)


# ============================================================
# 17. MODEL AGREEMENT BY ATTACK CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("MODEL AGREEMENT BY ATTACK CATEGORY")
print("=" * 70)


agreement_rows = []


for category in sorted(
    analysis_df["attack_cat"].unique()
):

    category_mask = (
        analysis_df["attack_cat"] == category
    )

    subset = analysis_df.loc[
        category_mask,
        [
            "MLP_prediction",
            "RandomForest_prediction",
            "XGBoost_prediction",
        ]
    ]

    unanimous = (
        subset.nunique(axis=1) == 1
    )

    agreement_rows.append({
        "attack_category": category,
        "sample_count": int(category_mask.sum()),
        "unanimous_prediction_count": int(
            unanimous.sum()
        ),
        "unanimous_prediction_rate": float(
            unanimous.mean()
        ),
    })


agreement_df = pd.DataFrame(
    agreement_rows
)


print(
    agreement_df.to_string(
        index=False
    )
)


agreement_df.to_csv(
    TABLE_DIR / "baseline_model_agreement_by_attack_category.csv",
    index=False
)


# ============================================================
# 18. SAVE COMPLETE DIAGNOSTIC METADATA
# ============================================================

diagnostic_metadata = {

    "experiment":
        "unsw_attack_category_diagnostic",

    "dataset":
        "UNSW-NB15",

    "training_file":
        str(TRAINING_FILE),

    "testing_file":
        str(TESTING_FILE),

    "models":
        list(loaded_models.keys()),

    "purpose":
        (
            "Post-hoc analysis of attack-category composition "
            "and clean baseline model errors."
        ),

    "attack_cat_used_as_model_feature":
        False,

    "models_retrained":
        False,

    "hyperparameters_changed":
        False,

    "test_set_modified":
        False,

    "adversarial_attacks_performed":
        False,

    "raw_training_rows":
        int(len(train_raw)),

    "raw_testing_rows":
        int(len(test_raw)),

    "processed_testing_rows":
        int(len(X_test)),

    "raw_processed_test_labels_identical":
        True,

    "outputs": {
        "category_comparison":
            "unsw_attack_category_train_test_comparison.csv",

        "training_crosstab":
            "unsw_training_attack_category_label_crosstab.csv",

        "testing_crosstab":
            "unsw_testing_attack_category_label_crosstab.csv",

        "false_negatives":
            "baseline_false_negatives_by_attack_category.csv",

        "false_positives":
            "baseline_false_positives_summary.csv",

        "category_errors":
            "baseline_error_rate_by_attack_category.csv",

        "model_agreement":
            "baseline_model_agreement_by_attack_category.csv",
    },
}


with open(
    LOG_DIR / "unsw_attack_category_diagnostic_metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        diagnostic_metadata,
        f,
        indent=4
    )


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ATTACK CATEGORY DIAGNOSTIC COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(TABLE_DIR)

print("\nMetadata saved to:")
print(LOG_DIR)

print("\nImportant:")
print("- No models were retrained.")
print("- No hyperparameters were changed.")
print("- No adversarial attacks were performed.")
print("- attack_cat was used only for post-hoc analysis.")
print("- The processed test set was not modified.")

print("\n" + "=" * 70)
print("NEXT STEP: REVIEW CATEGORY-LEVEL RESULTS")
print("=" * 70)