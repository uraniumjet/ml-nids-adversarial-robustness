from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

TARGET = "label"

DROP_COLUMNS = [
    "id",
    "attack_cat",
]


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print("Loading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training data: {train_df.shape}")
print(f"Testing data : {test_df.shape}")


# ---------------------------------------------------------------------
# Validate schema
# ---------------------------------------------------------------------

required_columns = set(DROP_COLUMNS + [TARGET])

for column in required_columns:
    if column not in train_df.columns:
        raise ValueError(f"Missing required training column: {column}")

    if column not in test_df.columns:
        raise ValueError(f"Missing required testing column: {column}")


# ---------------------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------------------

X_full = train_df.drop(columns=[TARGET] + DROP_COLUMNS)
y_full = train_df[TARGET].astype(int)

X_test = test_df.drop(columns=[TARGET] + DROP_COLUMNS)
y_test = test_df[TARGET].astype(int)


# ---------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_full,
    y_full,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_full,
)


print("\nSplit sizes:")
print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Testing    : {X_test.shape}")


# ---------------------------------------------------------------------
# Identify feature types
# ---------------------------------------------------------------------

categorical_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]


print("\nCategorical features:")
for feature in categorical_features:
    print(f"  - {feature}")

print("\nNumerical features:")
for feature in numerical_features:
    print(f"  - {feature}")


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)


# ---------------------------------------------------------------------
# FIT ONLY ON TRAINING DATA
# ---------------------------------------------------------------------

print("\nFitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)


# ---------------------------------------------------------------------
# Convert to DataFrames
# ---------------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index,
)

X_val_processed = pd.DataFrame(
    X_val_processed,
    columns=feature_names,
    index=X_val.index,
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index,
)


# ---------------------------------------------------------------------
# Save processed data
# ---------------------------------------------------------------------

X_train_processed.to_csv(
    PROCESSED_DIR / "unsw_X_train.csv",
    index=False,
)

X_val_processed.to_csv(
    PROCESSED_DIR / "unsw_X_val.csv",
    index=False,
)

X_test_processed.to_csv(
    PROCESSED_DIR / "unsw_X_test.csv",
    index=False,
)

y_train.to_csv(
    PROCESSED_DIR / "unsw_y_train.csv",
    index=False,
)

y_val.to_csv(
    PROCESSED_DIR / "unsw_y_val.csv",
    index=False,
)

y_test.to_csv(
    PROCESSED_DIR / "unsw_y_test.csv",
    index=False,
)


# ---------------------------------------------------------------------
# Save preprocessing pipeline
# ---------------------------------------------------------------------

joblib.dump(
    preprocessor,
    PROCESSED_DIR / "unsw_preprocessor.joblib",
)


# ---------------------------------------------------------------------
# Save metadata
# ---------------------------------------------------------------------

metadata = {
    "dataset": "UNSW-NB15",
    "random_state": RANDOM_STATE,
    "validation_size": VALIDATION_SIZE,
    "target": TARGET,
    "dropped_columns": DROP_COLUMNS,
    "original_training_shape": list(train_df.shape),
    "original_testing_shape": list(test_df.shape),
    "training_shape": list(X_train.shape),
    "validation_shape": list(X_val.shape),
    "testing_shape": list(X_test.shape),
    "processed_training_shape": list(X_train_processed.shape),
    "processed_validation_shape": list(X_val_processed.shape),
    "processed_testing_shape": list(X_test_processed.shape),
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "class_distribution": {
        "train": y_train.value_counts().sort_index().to_dict(),
        "validation": y_val.value_counts().sort_index().to_dict(),
        "test": y_test.value_counts().sort_index().to_dict(),
    },
}

with open(
    LOG_DIR / "unsw_preprocessing_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Processed features: {len(feature_names)}")

print("\nProcessed shapes:")
print(f"Training   : {X_train_processed.shape}")
print(f"Validation : {X_val_processed.shape}")
print(f"Testing    : {X_test_processed.shape}")

print("\nClass distributions:")

print("Training:")
print(y_train.value_counts().sort_index())

print("\nValidation:")
print(y_val.value_counts().sort_index())

print("\nTesting:")
print(y_test.value_counts().sort_index())

print("\nSaved:")
print("  data/processed/unsw_X_train.csv")
print("  data/processed/unsw_X_val.csv")
print("  data/processed/unsw_X_test.csv")
print("  data/processed/unsw_y_train.csv")
print("  data/processed/unsw_y_val.csv")
print("  data/processed/unsw_y_test.csv")
print("  data/processed/unsw_preprocessor.joblib")
print("  results/logs/unsw_preprocessing_metadata.json")
