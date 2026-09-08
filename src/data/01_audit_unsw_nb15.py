from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# UNSW-NB15 DATASET AUDIT
# ============================================================
# Purpose:
#   Inspect the raw UNSW-NB15 training/testing files without
#   modifying them.
#
# Output:
#   - Console summary
#   - JSON audit report
#
# Raw data is READ ONLY.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "results" / "logs"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"
FEATURE_FILE = RAW_DIR / "NUSW-NB15_features.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_dataframe(df, name):
    """Return a structured audit of a dataframe."""

    report = {
        "name": name,
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
            if value > 0
        },
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "infinite_values": {},
        "categorical_columns": {},
        "numerical_ranges": {},
        "unique_counts": {
            column: int(df[column].nunique(dropna=False))
            for column in df.columns
        },
    }

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------
    numeric_df = df.select_dtypes(include=[np.number])

    if not numeric_df.empty:
        infinite_counts = np.isinf(numeric_df).sum()

        report["infinite_values"] = {
            column: int(value)
            for column, value in infinite_counts.items()
            if value > 0
        }

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------
    categorical_df = df.select_dtypes(
        include=["object", "category"]
    )

    for column in categorical_df.columns:
        values = df[column].value_counts(
            dropna=False
        )

        report["categorical_columns"][column] = {
            "unique_values": int(df[column].nunique(dropna=False)),
            "value_counts": {
                str(index): int(value)
                for index, value in values.items()
            },
        }

    # --------------------------------------------------------
    # Numerical ranges
    # --------------------------------------------------------
    for column in numeric_df.columns:
        series = numeric_df[column]

        report["numerical_ranges"][column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
        }

    return report


def inspect_feature_definitions(path):
    """Inspect the UNSW feature-definition file."""

    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
        }

    try:
        features = pd.read_csv(path, encoding="latin1")
    except Exception as exc:
        return {
            "status": "error",
            "path": str(path),
            "error": str(exc),
        }

    return {
        "status": "ok",
        "path": str(path),
        "shape": {
            "rows": int(features.shape[0]),
            "columns": int(features.shape[1]),
        },
        "columns": list(features.columns),
        "records": features.fillna("").astype(str).to_dict(
            orient="records"
        ),
    }


def identify_special_columns(df):
    """Identify likely identifiers, targets and leakage-related fields."""

    columns = set(df.columns)

    identifier_candidates = [
        column
        for column in ["id", "srcip", "dstip", "sport", "dsport"]
        if column in columns
    ]

    target_candidates = [
        column
        for column in ["label", "attack_cat"]
        if column in columns
    ]

    possible_derived_features = [
        column
        for column in [
            "ct_srv_src",
            "ct_state_ttl",
            "ct_dst_ltm",
            "ct_src_dport_ltm",
            "ct_dst_sport_ltm",
            "ct_dst_src_ltm",
            "ct_src_ltm",
            "ct_srv_dst",
            "is_sm_ips_ports",
            "is_ftp_login",
            "ct_ftp_cmd",
            "ct_flw_http_mthd",
        ]
        if column in columns
    ]

    return {
        "identifier_candidates": identifier_candidates,
        "target_candidates": target_candidates,
        "possible_derived_or_contextual_features": (
            possible_derived_features
        ),
    }


def audit_dataset():

    print("=" * 70)
    print("UNSW-NB15 DATASET AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------
    files = {
        "training": TRAIN_FILE,
        "testing": TEST_FILE,
        "feature_definitions": FEATURE_FILE,
    }

    print("\nFile availability:")

    for name, path in files.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"  {name:20s}: {status} -> {path}")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_FILE}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------
    print("\nLoading training dataset...")
    train = pd.read_csv(TRAIN_FILE)

    print("Loading testing dataset...")
    test = pd.read_csv(TEST_FILE)

    print("\nDatasets loaded successfully.")

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASIC DATASET INFORMATION")
    print("=" * 70)

    print(f"\nTraining shape: {train.shape}")
    print(f"Testing shape : {test.shape}")

    print("\nTraining columns:")
    for i, column in enumerate(train.columns, start=1):
        print(f"  {i:2d}. {column}")

    print("\nTesting columns:")
    for i, column in enumerate(test.columns, start=1):
        print(f"  {i:2d}. {column}")

    # --------------------------------------------------------
    # Column consistency
    # --------------------------------------------------------
    columns_match = list(train.columns) == list(test.columns)

    print("\nTraining/testing columns identical:", columns_match)

    if not columns_match:
        train_only = [
            c for c in train.columns
            if c not in test.columns
        ]

        test_only = [
            c for c in test.columns
            if c not in train.columns
        ]

        print("Training-only columns:", train_only)
        print("Testing-only columns:", test_only)

    # --------------------------------------------------------
    # DataFrame audits
    # --------------------------------------------------------
    print("\nRunning training audit...")
    training_report = inspect_dataframe(
        train,
        "training",
    )

    print("Running testing audit...")
    testing_report = inspect_dataframe(
        test,
        "testing",
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for dataset_name, df in [
        ("Training", train),
        ("Testing", test),
    ]:

        print(f"\n{dataset_name} label distribution:")

        if "label" in df.columns:
            print(
                df["label"]
                .value_counts(dropna=False)
                .sort_index()
            )

        if "attack_cat" in df.columns:
            print(
                f"\n{dataset_name} attack category distribution:"
            )

            print(
                df["attack_cat"]
                .value_counts(dropna=False)
            )

    # --------------------------------------------------------
    # Special columns
    # --------------------------------------------------------
    special_columns = identify_special_columns(train)

    print("\n" + "=" * 70)
    print("SPECIAL / POTENTIALLY IMPORTANT COLUMNS")
    print("=" * 70)

    print(
        "\nIdentifier candidates:",
        special_columns["identifier_candidates"],
    )

    print(
        "\nTarget candidates:",
        special_columns["target_candidates"],
    )

    print(
        "\nPossible derived/contextual features:"
    )

    for column in special_columns[
        "possible_derived_or_contextual_features"
    ]:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Feature definitions
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("FEATURE-DEFINITION FILE")
    print("=" * 70)

    feature_report = inspect_feature_definitions(
        FEATURE_FILE
    )

    print(
        "Status:",
        feature_report.get("status"),
    )

    if feature_report.get("status") == "ok":
        print(
            "Feature-definition shape:",
            feature_report["shape"],
        )

        print(
            "Feature-definition columns:",
            feature_report["columns"],
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------
    report = {
        "dataset": "UNSW-NB15",
        "raw_directory": str(RAW_DIR),
        "files": {
            name: {
                "path": str(path),
                "exists": path.exists(),
            }
            for name, path in files.items()
        },
        "training": training_report,
        "testing": testing_report,
        "columns_match": columns_match,
        "special_columns": special_columns,
        "feature_definitions": feature_report,
    }

    output_file = REPORT_DIR / "unsw_nb15_audit.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"\nMachine-readable report saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    audit_dataset()