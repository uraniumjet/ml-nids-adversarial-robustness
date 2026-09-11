"""
Full-test trustworthiness analysis for adversarial ML-NIDS evaluation.

Purpose:
    Recalculate Brier score and Expected Calibration Error (ECE)
    on the complete UNSW-NB15 test set.

Design:
    - Full test set: 82,332 observations.
    - The same 1,000 attack-eligible malicious observations used in
      the adversarial experiments are attacked.
    - All remaining test observations remain unchanged.
    - Conventional and constraint-aware attacks use the same
      experimental settings as the main attack experiments.

This analysis provides a full-distribution calibration assessment.
The earlier 1,000-sample analysis is preserved separately.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

X_TEST_PATH = ROOT / "data/processed/unsw_X_test.csv"
Y_TEST_PATH = ROOT / "data/processed/unsw_y_test.csv"

X_TRAIN_PATH = ROOT / "data/processed/unsw_X_train.csv"

MODEL_PATH = ROOT / "results/models/mlp_clean.joblib"

SAMPLE_PATH = (
    ROOT
    / "results/attacks/conventional_attack_sample_indices.csv"
)

ATTACK_MODULE_PATH = (
    ROOT
    / "src/attacks/06_conventional_attacks.py"
)

OUTPUT_DIR = ROOT / "results/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = (
    ROOT / "data/processed/unsw_preprocessor.joblib"
)

METADATA_PATH = (
    ROOT / "results/logs/unsw_preprocessing_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

EPSILONS = [0.05, 0.10, 0.20]

PGD_STEPS = 10

N_ECE_BINS = 10


# ============================================================
# LOAD ATTACK MODULE
# ============================================================

spec = importlib.util.spec_from_file_location(
    "conventional_attacks",
    ATTACK_MODULE_PATH
)

conventional_attacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conventional_attacks)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FULL-TEST TRUSTWORTHINESS ANALYSIS")
print("=" * 70)

X_test = pd.read_csv(X_TEST_PATH).values
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

X_train = pd.read_csv(X_TRAIN_PATH).values

model = joblib.load(MODEL_PATH)

sample_indices = (
    pd.read_csv(SAMPLE_PATH)["test_index"].values
)

sample_indices = np.asarray(
    sample_indices,
    dtype=int
)

print(f"Full test samples: {len(X_test)}")
print(f"Attack-eligible samples: {len(sample_indices)}")
print(f"Processed features: {X_test.shape[1]}")


# ============================================================
# VERIFY ATTACK SAMPLE
# ============================================================

if len(sample_indices) != 1000:
    raise ValueError(
        "Expected exactly 1,000 attack-eligible samples."
    )

if len(np.unique(sample_indices)) != len(sample_indices):
    raise ValueError(
        "Attack sample indices contain duplicates."
    )

if np.any(sample_indices < 0) or np.any(
    sample_indices >= len(X_test)
):
    raise ValueError(
        "Attack sample indices contain invalid test indices."
    )

if not np.all(y_test[sample_indices] == 1):
    raise ValueError(
        "Attack sample contains non-malicious observations."
    )


# ============================================================
# FEATURE MASK
# ============================================================

feature_mask = (
    conventional_attacks.get_numerical_feature_mask(
        n_features=X_test.shape[1]
    )
)


# ============================================================
# CONSTRAINT INFORMATION
# ============================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

numerical_features = metadata["numerical_features"]

feature_names = [
    f"numerical__{name}"
    for name in numerical_features
]

encoder = preprocessor.named_transformers_["categorical"]

encoded_feature_names = (
    encoder.get_feature_names_out(
        metadata["categorical_features"]
    )
)

feature_names.extend(
    f"categorical__{name}"
    for name in encoded_feature_names
)

constraint_mask = feature_mask.copy()

BINARY_FIXED = {
    "numerical__is_ftp_login",
    "numerical__is_sm_ips_ports",
}

for i, name in enumerate(feature_names):

    if name in BINARY_FIXED:
        constraint_mask[i] = False

    if name.startswith("numerical__ct_"):
        constraint_mask[i] = False

    if name.startswith("categorical__"):
        constraint_mask[i] = False


# ============================================================
# TRAINING-DERIVED BOUNDS
# ============================================================

train_lower = np.min(
    X_train,
    axis=0
)

train_upper = np.max(
    X_train,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

def get_probability(X):
    return (
        conventional_attacks
        .get_positive_class_probability(
            model,
            X
        )
    )


# ============================================================
# CALIBRATION METRICS
# ============================================================

def brier_score(
    y_true,
    probabilities,
):
    """
    Mean squared probability error.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    return np.mean(
        (probabilities - y_true) ** 2
    )


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):
    """
    Equal-width Expected Calibration Error.

    ECE is calculated over the complete test distribution.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    bin_records = []

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (
                (probabilities >= bins[i])
                & (probabilities <= bins[i + 1])
            )

        else:

            mask = (
                (probabilities >= bins[i])
                & (probabilities < bins[i + 1])
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        accuracy = np.mean(
            y_true[mask]
        )

        confidence = np.mean(
            probabilities[mask]
        )

        weight = count / len(y_true)

        gap = abs(
            accuracy - confidence
        )

        ece += weight * gap

        bin_records.append({
            "bin": i,
            "lower": bins[i],
            "upper": bins[i + 1],
            "count": count,
            "accuracy": accuracy,
            "confidence": confidence,
            "gap": gap,
        })

    return ece, pd.DataFrame(bin_records)


# ============================================================
# CONSTRAINT APPLICATION
# ============================================================

def apply_constraints(
    X_clean,
    X_candidate,
):
    """
    Restore fixed features and enforce empirical
    training-derived bounds.
    """

    X_constrained = X_candidate.copy()

    X_constrained[:, ~constraint_mask] = (
        X_clean[:, ~constraint_mask]
    )

    for i in np.where(constraint_mask)[0]:

        X_constrained[:, i] = np.clip(
            X_constrained[:, i],
            train_lower[i],
            train_upper[i],
        )

    return X_constrained


# ============================================================
# ATTACK RECONSTRUCTION
# ============================================================

X_attack = X_test[sample_indices].copy()
y_attack = y_test[sample_indices].copy()


def generate_attack(
    attack_type,
    epsilon,
):
    """
    Reconstruct the same attack conditions used in the
    trustworthiness experiment.
    """

    if attack_type == "FGSM":

        return conventional_attacks.fgsm_attack_mlp(
            model=model,
            X=X_attack,
            y=y_attack,
            epsilon=epsilon,
            feature_mask=feature_mask,
        )

    if attack_type == "PGD":

        return conventional_attacks.pgd_attack_mlp(
            model=model,
            X=X_attack,
            y=y_attack,
            epsilon=epsilon,
            step_size=epsilon / 4,
            num_steps=PGD_STEPS,
            feature_mask=feature_mask,
        )

    if attack_type == "constraint_aware_FGSM":

        X_adv = conventional_attacks.fgsm_attack_mlp(
            model=model,
            X=X_attack,
            y=y_attack,
            epsilon=epsilon,
            feature_mask=constraint_mask,
        )

        return apply_constraints(
            X_attack,
            X_adv
        )

    if attack_type == "constraint_aware_PGD":

        lower = np.maximum(
            X_attack - epsilon,
            train_lower
        )

        upper = np.minimum(
            X_attack + epsilon,
            train_upper
        )

        X_adv = conventional_attacks.pgd_attack_mlp(
            model=model,
            X=X_attack,
            y=y_attack,
            epsilon=epsilon,
            step_size=epsilon / 4,
            num_steps=PGD_STEPS,
            feature_mask=constraint_mask,
            lower_bounds=lower,
            upper_bounds=upper,
        )

        return apply_constraints(
            X_attack,
            X_adv
        )

    raise ValueError(
        f"Unknown attack type: {attack_type}"
    )


# ============================================================
# FULL-TEST CONDITION BUILDER
# ============================================================

def build_full_test_condition(
    X_adv_attack,
):
    """
    Start with the complete clean test set and replace only
    the 1,000 attack-eligible observations.
    """

    X_condition = X_test.copy()

    X_condition[sample_indices] = X_adv_attack

    return X_condition


# ============================================================
# CLEAN FULL-TEST BASELINE
# ============================================================

print("\nCalculating clean full-test predictions...")

clean_probability = get_probability(
    X_test
)

clean_prediction = (
    clean_probability >= 0.5
).astype(int)

clean_brier = brier_score(
    y_test,
    clean_probability
)

clean_ece, clean_bins = (
    expected_calibration_error(
        y_test,
        clean_probability,
        n_bins=N_ECE_BINS,
    )
)

print(
    f"Clean Brier: {clean_brier:.6f}"
)

print(
    f"Clean ECE:   {clean_ece:.6f}"
)


# ============================================================
# RESULTS
# ============================================================

results = []

clean_result = {
    "model": "mlp",
    "condition": "clean",
    "epsilon": 0.0,
    "test_samples": len(y_test),
    "attacked_samples": 0,

    "successful_evasions": 0,
    "attack_success_rate": 0.0,

    "clean_brier_score": clean_brier,
    "attacked_brier_score": clean_brier,
    "brier_score_change": 0.0,

    "clean_ece": clean_ece,
    "attacked_ece": clean_ece,
    "ece_change": 0.0,

    "mean_test_probability": np.mean(
        clean_probability
    ),

    "mean_probability_change": 0.0,

    "prediction_changes": 0,

    "prediction_change_rate": 0.0,
}

results.append(clean_result)


# ============================================================
# ATTACK CONDITIONS
# ============================================================

conditions = [
    ("FGSM", "conventional"),
    ("PGD", "conventional"),
    ("constraint_aware_FGSM", "constraint_aware"),
    ("constraint_aware_PGD", "constraint_aware"),
]


for attack_type, category in conditions:

    for epsilon in EPSILONS:

        print("\n" + "=" * 70)
        print(
            f"{attack_type} | "
            f"epsilon={epsilon}"
        )
        print("=" * 70)

        X_adv_attack = generate_attack(
            attack_type,
            epsilon
        )

        X_full_adv = build_full_test_condition(
            X_adv_attack
        )

        attacked_probability = get_probability(
            X_full_adv
        )

        attacked_prediction = (
            attacked_probability >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Evasion on the predefined attack-eligible subset
        # ----------------------------------------------------

        clean_attack_probability = (
            clean_probability[sample_indices]
        )

        attacked_attack_probability = (
            attacked_probability[sample_indices]
        )

        clean_attack_prediction = (
            clean_attack_probability >= 0.5
        ).astype(int)

        attacked_attack_prediction = (
            attacked_attack_probability >= 0.5
        ).astype(int)

        successful_evasion = (
            (y_attack == 1)
            & (clean_attack_prediction == 1)
            & (attacked_attack_prediction == 0)
        )

        asr = (
            np.sum(successful_evasion)
            / len(y_attack)
        )

        # ----------------------------------------------------
        # Full-test trustworthiness
        # ----------------------------------------------------

        attacked_brier = brier_score(
            y_test,
            attacked_probability
        )

        attacked_ece, _ = (
            expected_calibration_error(
                y_test,
                attacked_probability,
                n_bins=N_ECE_BINS,
            )
        )

        probability_change = (
            attacked_probability -
            clean_probability
        )

        prediction_changes = np.sum(
            attacked_prediction != clean_prediction
        )

        result = {
            "model": "mlp",
            "condition": attack_type,
            "epsilon": epsilon,
            "test_samples": len(y_test),
            "attacked_samples": len(sample_indices),

            "successful_evasions": int(
                successful_evasion.sum()
            ),

            "attack_success_rate": asr,

            "clean_brier_score": clean_brier,

            "attacked_brier_score": (
                attacked_brier
            ),

            "brier_score_change": (
                attacked_brier -
                clean_brier
            ),

            "clean_ece": clean_ece,

            "attacked_ece": attacked_ece,

            "ece_change": (
                attacked_ece -
                clean_ece
            ),

            "mean_test_probability": (
                np.mean(attacked_probability)
            ),

            "mean_probability_change": (
                np.mean(probability_change)
            ),

            "prediction_changes": int(
                prediction_changes
            ),

            "prediction_change_rate": (
                prediction_changes /
                len(y_test)
            ),
        }

        results.append(result)

        print(
            f"ASR: {asr:.4f}"
        )

        print(
            f"Full-test Brier: "
            f"{clean_brier:.6f} -> "
            f"{attacked_brier:.6f}"
        )

        print(
            f"Full-test ECE: "
            f"{clean_ece:.6f} -> "
            f"{attacked_ece:.6f}"
        )

        print(
            f"ECE change: "
            f"{attacked_ece - clean_ece:.6f}"
        )

        print(
            f"Prediction changes: "
            f"{prediction_changes}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR /
    "full_test_trustworthiness_results_mlp.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE CLEAN CALIBRATION BINS
# ============================================================

bins_path = (
    OUTPUT_DIR /
    "full_test_clean_calibration_bins_mlp.csv"
)

clean_bins.to_csv(
    bins_path,
    index=False
)


# ============================================================
# SAVE METADATA
# ============================================================

analysis_metadata = {
    "model": "mlp",
    "dataset": "UNSW-NB15",

    "seed": SEED,

    "full_test_samples": int(
        len(y_test)
    ),

    "attack_eligible_samples": int(
        len(sample_indices)
    ),

    "ece_bins": N_ECE_BINS,

    "attack_eligible_selection": (
        "same 1,000 malicious clean-correct "
        "test observations used by the main "
        "adversarial experiments"
    ),

    "unattacked_test_observations": int(
        len(y_test) - len(sample_indices)
    ),

    "calibration_scope": (
        "complete UNSW-NB15 test distribution"
    ),

    "constraint_type": (
        "constraint-aware feature-space"
    ),

    "host_space_attack": False,

    "training_bounds": (
        "empirical minimum and maximum values "
        "computed from the training split only"
    ),

    "purpose": (
        "full-distribution trustworthiness audit; "
        "the previous 1,000-sample analysis is "
        "preserved separately"
    ),
}

metadata_output_path = (
    OUTPUT_DIR /
    "full_test_trustworthiness_metadata_mlp.json"
)

with open(
    metadata_output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        analysis_metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FULL-TEST TRUSTWORTHINESS ANALYSIS COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "condition",
            "epsilon",
            "attack_success_rate",
            "clean_brier_score",
            "attacked_brier_score",
            "brier_score_change",
            "clean_ece",
            "attacked_ece",
            "ece_change",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(results_path)
print(bins_path)
print(metadata_output_path)