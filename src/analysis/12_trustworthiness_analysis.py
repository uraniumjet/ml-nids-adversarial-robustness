"""
Trustworthiness analysis for adversarial ML-NIDS evaluation.

Primary trustworthiness metrics:
    - Brier score
    - Expected Calibration Error (ECE)

Secondary paired attack metrics:
    - confidence statistics
    - confidence drop
    - prediction changes
    - attack success rate

Conditions:
    - Clean
    - Conventional FGSM
    - Conventional PGD
    - Constraint-aware FGSM
    - Constraint-aware PGD

IMPORTANT:
The adversarial attacks are generated only for the fixed 1,000
clean-correct malicious test observations used in the robustness
experiments.

For trustworthiness/calibration, however, adversarial probabilities
are inserted back into the COMPLETE 82,332-observation test set.
Therefore:

    - Full-test Brier/ECE = PRIMARY trustworthiness analysis
    - 1,000-sample paired confidence/ASR analysis = SECONDARY analysis

The original subset-level analysis is preserved conceptually but
is no longer used as the primary calibration result.
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
    ROOT /
    "results/attacks/"
    "conventional_attack_sample_indices.csv"
)

ATTACK_MODULE_PATH = (
    ROOT /
    "src/attacks/"
    "06_conventional_attacks.py"
)

OUTPUT_DIR = ROOT / "results/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

EPSILONS = [0.05, 0.10, 0.20]

PGD_STEPS = 10

ECE_BINS = 10


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

X_attack = X_test[sample_indices].copy()
y_attack = y_test[sample_indices].copy()

print(f"Full test samples: {len(X_test)}")
print(f"Attack subset samples: {len(X_attack)}")
print(f"Processed features: {X_test.shape[1]}")


# ============================================================
# SANITY CHECKS
# ============================================================

assert len(X_test) == len(y_test)

assert np.all(
    y_attack == 1
), "Attack subset must contain only malicious observations."

assert len(np.unique(sample_indices)) == len(sample_indices)

assert np.all(
    sample_indices >= 0
)

assert np.all(
    sample_indices < len(X_test)
)


# ============================================================
# FEATURE MASKS
# ============================================================

feature_mask = conventional_attacks.get_numerical_feature_mask(
    n_features=X_test.shape[1]
)


# ============================================================
# LOAD CONSTRAINT INFORMATION
# ============================================================

with open(
    ROOT / "results/logs/unsw_preprocessing_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

preprocessor = joblib.load(
    ROOT / "data/processed/unsw_preprocessor.joblib"
)

numerical_features = metadata["numerical_features"]

feature_names = [
    f"numerical__{name}"
    for name in numerical_features
]

encoder = preprocessor.named_transformers_["categorical"]

encoded_feature_names = encoder.get_feature_names_out(
    metadata["categorical_features"]
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
# CALIBRATION FUNCTIONS
# ============================================================

def brier_score(y_true, probabilities):
    """
    Binary Brier score.
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
    Expected Calibration Error using equal-width
    probability bins over [0, 1].
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
                (probabilities >= bins[i]) &
                (probabilities <= bins[i + 1])
            )
        else:
            mask = (
                (probabilities >= bins[i]) &
                (probabilities < bins[i + 1])
            )

        count = np.sum(mask)

        if count == 0:
            continue

        accuracy = np.mean(
            y_true[mask]
        )

        confidence = np.mean(
            probabilities[mask]
        )

        weight = count / len(y_true)

        calibration_gap = abs(
            accuracy - confidence
        )

        ece += (
            weight *
            calibration_gap
        )

        bin_records.append({
            "bin": i,
            "lower": bins[i],
            "upper": bins[i + 1],
            "count": int(count),
            "accuracy": accuracy,
            "confidence": confidence,
            "gap": calibration_gap,
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
    Restore fixed features and enforce training-derived
    empirical bounds for eligible features.
    """

    X_constrained = X_candidate.copy()

    # Restore fixed features.
    X_constrained[:, ~constraint_mask] = (
        X_clean[:, ~constraint_mask]
    )

    # Enforce training-derived empirical bounds.
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

def generate_attack(
    attack_type,
    epsilon,
):
    """
    Reconstruct adversarial examples using the same
    experimental settings as the attack experiments.
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
# PREDICTION HELPER
# ============================================================

def get_probability(X):
    return conventional_attacks.get_positive_class_probability(
        model,
        X
    )


# ============================================================
# CLEAN FULL-TEST PREDICTIONS
# ============================================================

clean_probability_full = get_probability(
    X_test
)

clean_prediction_full = (
    clean_probability_full >= 0.5
).astype(int)


# ============================================================
# CLEAN ATTACK-SUBSET PREDICTIONS
# ============================================================

clean_probability_attack = (
    clean_probability_full[sample_indices]
)

clean_prediction_attack = (
    clean_probability_attack >= 0.5
).astype(int)


# ============================================================
# VERIFY ATTACK ELIGIBILITY
# ============================================================

clean_correct_malicious = (
    (y_attack == 1) &
    (clean_prediction_attack == 1)
)

if not np.all(clean_correct_malicious):

    raise RuntimeError(
        "The saved attack sample set contains observations "
        "that are not clean-correct malicious samples."
    )


# ============================================================
# CLEAN FULL-TEST TRUSTWORTHINESS
# ============================================================

clean_brier_full = brier_score(
    y_test,
    clean_probability_full
)

clean_ece_full, clean_bins_full = (
    expected_calibration_error(
        y_test,
        clean_probability_full,
        n_bins=ECE_BINS
    )
)

print("\nClean full-test metrics:")
print(
    f"  Brier: {clean_brier_full:.6f}"
)

print(
    f"  ECE:   {clean_ece_full:.6f}"
)


# ============================================================
# RESULT STORAGE
# ============================================================

full_test_results = []

paired_results = []


# ============================================================
# CLEAN RESULT
# ============================================================

full_test_results.append({
    "model": "mlp",
    "condition": "clean",
    "epsilon": 0.0,
    "test_samples": len(y_test),
    "attack_subset_samples": len(y_attack),

    "brier_score": clean_brier_full,
    "ece": clean_ece_full,

    "brier_change_from_clean": 0.0,
    "ece_change_from_clean": 0.0,
})


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
            f"{attack_type} | epsilon={epsilon}"
        )
        print("=" * 70)

        X_adv = generate_attack(
            attack_type,
            epsilon
        )

        attacked_probability = get_probability(
            X_adv
        )

        attacked_prediction = (
            attacked_probability >= 0.5
        ).astype(int)


        # ----------------------------------------------------
        # PAIRED ATTACK ANALYSIS
        # ----------------------------------------------------

        successful_evasion = (
            (y_attack == 1) &
            (clean_prediction_attack == 1) &
            (attacked_prediction == 0)
        )

        confidence_drop = (
            clean_probability_attack -
            attacked_probability
        )

        prediction_changes = (
            clean_prediction_attack !=
            attacked_prediction
        )


        attack_success_rate = (
            successful_evasion.sum() /
            len(y_attack)
        )


        # ----------------------------------------------------
        # FULL-TEST ADVERSARIAL PROBABILITIES
        # ----------------------------------------------------

        adversarial_probability_full = (
            clean_probability_full.copy()
        )

        adversarial_probability_full[
            sample_indices
        ] = attacked_probability


        # ----------------------------------------------------
        # FULL-TEST PREDICTIONS
        # ----------------------------------------------------

        adversarial_prediction_full = (
            adversarial_probability_full >= 0.5
        ).astype(int)


        # ----------------------------------------------------
        # FULL-TEST TRUSTWORTHINESS
        # ----------------------------------------------------

        adversarial_brier_full = brier_score(
            y_test,
            adversarial_probability_full
        )

        adversarial_ece_full, _ = (
            expected_calibration_error(
                y_test,
                adversarial_probability_full,
                n_bins=ECE_BINS
            )
        )


        # ----------------------------------------------------
        # STORE FULL-TEST RESULT
        # ----------------------------------------------------

        full_test_results.append({
            "model": "mlp",
            "condition": attack_type,
            "epsilon": epsilon,
            "test_samples": len(y_test),
            "attack_subset_samples": len(y_attack),

            "brier_score": adversarial_brier_full,
            "ece": adversarial_ece_full,

            "brier_change_from_clean": (
                adversarial_brier_full -
                clean_brier_full
            ),

            "ece_change_from_clean": (
                adversarial_ece_full -
                clean_ece_full
            ),
        })


        # ----------------------------------------------------
        # STORE PAIRED RESULT
        # ----------------------------------------------------

        paired_results.append({
            "model": "mlp",
            "condition": attack_type,
            "epsilon": epsilon,
            "samples": len(y_attack),

            "successful_evasions": int(
                successful_evasion.sum()
            ),

            "attack_success_rate": (
                attack_success_rate
            ),

            "mean_clean_probability": (
                np.mean(clean_probability_attack)
            ),

            "mean_attacked_probability": (
                np.mean(attacked_probability)
            ),

            "mean_confidence_drop": (
                np.mean(confidence_drop)
            ),

            "median_confidence_drop": (
                np.median(confidence_drop)
            ),

            "prediction_changes": int(
                prediction_changes.sum()
            ),

            "prediction_change_rate": (
                prediction_changes.mean()
            ),
        })


        # ----------------------------------------------------
        # CONSOLE OUTPUT
        # ----------------------------------------------------

        print(
            f"ASR: {attack_success_rate:.4f}"
        )

        print(
            f"Mean confidence drop: "
            f"{np.mean(confidence_drop):.4f}"
        )

        print(
            f"Full-test Brier: "
            f"{clean_brier_full:.6f} "
            f"-> "
            f"{adversarial_brier_full:.6f}"
        )

        print(
            f"Full-test ECE: "
            f"{clean_ece_full:.6f} "
            f"-> "
            f"{adversarial_ece_full:.6f}"
        )


# ============================================================
# SAVE FULL-TEST RESULTS
# ============================================================

full_test_results_df = pd.DataFrame(
    full_test_results
)

full_test_path = (
    OUTPUT_DIR /
    "trustworthiness_full_test_results_mlp.csv"
)

full_test_results_df.to_csv(
    full_test_path,
    index=False
)


# ============================================================
# SAVE PAIRED ATTACK RESULTS
# ============================================================

paired_results_df = pd.DataFrame(
    paired_results
)

paired_path = (
    OUTPUT_DIR /
    "trustworthiness_paired_attack_results_mlp.csv"
)

paired_results_df.to_csv(
    paired_path,
    index=False
)


# ============================================================
# SAVE CLEAN CALIBRATION BINS
# ============================================================

clean_bins_path = (
    OUTPUT_DIR /
    "clean_full_test_calibration_bins_mlp.csv"
)

clean_bins_full.to_csv(
    clean_bins_path,
    index=False
)


# ============================================================
# SAVE METADATA
# ============================================================

analysis_metadata = {
    "model": "mlp",
    "dataset": "UNSW-NB15",

    "seed": SEED,

    "full_test_samples": int(len(y_test)),

    "attack_subset_samples": int(len(y_attack)),

    "attack_subset_definition": (
        "malicious observations correctly classified "
        "under clean conditions"
    ),

    "epsilon_values": EPSILONS,

    "pgd_steps": PGD_STEPS,

    "pgd_step_size": (
        "epsilon / 4"
    ),

    "ece_bins": ECE_BINS,

    "ece_binning": (
        "equal-width bins over [0, 1]"
    ),

    "primary_trustworthiness_population": (
        "complete UNSW-NB15 test set"
    ),

    "paired_attack_population": (
        "fixed 1000-sample malicious clean-correct subset"
    ),

    "full_test_construction": (
        "clean probabilities for all test observations; "
        "adversarial probabilities replace clean probabilities "
        "only at attacked sample indices"
    ),

    "constraint_type": (
        "constraint-aware feature-space"
    ),

    "host_space_attack": False,

    "training_derived_bounds": True,

    "fixed_features_preserved": True,

    "categorical_features_perturbed": False,
}


metadata_path = (
    OUTPUT_DIR /
    "trustworthiness_full_test_metadata_mlp.json"
)

with open(
    metadata_path,
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

print("\nPRIMARY FULL-TEST RESULTS")

print(
    full_test_results_df[
        [
            "condition",
            "epsilon",
            "test_samples",
            "brier_score",
            "brier_change_from_clean",
            "ece",
            "ece_change_from_clean",
        ]
    ].to_string(index=False)
)

print("\nSECONDARY PAIRED ATTACK RESULTS")

print(
    paired_results_df[
        [
            "condition",
            "epsilon",
            "attack_success_rate",
            "mean_confidence_drop",
            "prediction_change_rate",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(full_test_path)
print(paired_path)
print(clean_bins_path)
print(metadata_path)