"""
Constraint-aware adversarial attack evaluation for UNSW-NB15.

This module implements constrained feature-space attacks.

Constraints:
    - Only selected numerical traffic features may be perturbed.
    - Categorical one-hot features remain fixed.
    - Binary/semantic features remain fixed.
    - Contextual/derived features remain fixed.
    - Perturbations remain inside the empirical training-data bounds.
    - The same L-infinity epsilon budgets are used as the conventional attacks.

This is a constraint-aware feature-space attack.
It is NOT a host-space attack.
"""

from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import joblib


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

X_TRAIN_PATH = ROOT / "data/processed/unsw_X_train.csv"
X_TEST_PATH = ROOT / "data/processed/unsw_X_test.csv"
Y_TEST_PATH = ROOT / "data/processed/unsw_y_test.csv"

MODEL_PATHS = {
    "mlp": ROOT / "results/models/mlp_clean.joblib",
}

ATTACK_MODULE_PATH = ROOT / "src/attacks/06_conventional_attacks.py"

OUTPUT_DIR = ROOT / "results/attacks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42
N_SAMPLES = 1000

EPSILONS = [0.05, 0.10, 0.20]

PGD_STEPS = 10


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
print("CONSTRAINT-AWARE ATTACK EXPERIMENT")
print("=" * 70)

X_train = pd.read_csv(X_TRAIN_PATH).values
X_test = pd.read_csv(X_TEST_PATH).values
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Processed features: {X_test.shape[1]}")


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

# Read feature names from the preprocessing metadata.
metadata_path = (
    ROOT / "results/logs/unsw_preprocessing_metadata.json"
)

import json

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ============================================================
# RECONSTRUCT PROCESSED FEATURE NAMES
# ============================================================

preprocessor_path = (
    ROOT / "data/processed/unsw_preprocessor.joblib"
)

preprocessor = joblib.load(preprocessor_path)

numerical_features = metadata["numerical_features"]
categorical_features = metadata["categorical_features"]

feature_names = [
    f"numerical__{name}"
    for name in numerical_features
]

encoder = preprocessor.named_transformers_["categorical"]   

encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
)

feature_names.extend(
    f"categorical__{name}"
    for name in encoded_feature_names
)

if len(feature_names) != X_test.shape[1]:
    raise ValueError(
        f"Reconstructed feature count "
        f"({len(feature_names)}) does not match "
        f"processed test feature count "
        f"({X_test.shape[1]})."
    )

print(
    f"Reconstructed processed feature names: "
    f"{len(feature_names)}"
)


# ============================================================
# START WITH THE 39 NUMERICAL FEATURES
# ============================================================

numerical_mask = np.array([
    name.startswith("numerical__")
    for name in feature_names
])


# ============================================================
# FROZEN NON-PERTURBABLE FEATURES
# ============================================================

# Binary / semantic features.
BINARY_FIXED = {
    "numerical__is_ftp_login",
    "numerical__is_sm_ips_ports",
}


# Contextual / derived features.
CONTEXTUAL_PREFIXES = (
    "numerical__ct_",
)


constraint_mask = numerical_mask.copy()


for i, name in enumerate(feature_names):

    # Binary / semantic features
    if name in BINARY_FIXED:
        constraint_mask[i] = False

    # Contextual / derived features
    if name.startswith(CONTEXTUAL_PREFIXES):
        constraint_mask[i] = False


    # Categorical one-hot features
    if name.startswith("categorical__"):
        constraint_mask[i] = False


# ============================================================
# SANITY CHECK FEATURE GROUPS
# ============================================================

print("\nFeature constraint summary")
print("-" * 70)

print(
    f"All processed features:       "
    f"{len(feature_names)}"
)

print(
    f"Numerical features:            "
    f"{numerical_mask.sum()}"
)

print(
    f"Constraint-eligible features:   "
    f"{constraint_mask.sum()}"
)

print(
    f"Fixed features:                 "
    f"{(~constraint_mask).sum()}"
)

print("\nConstraint-eligible features:")

for i, name in enumerate(feature_names):
    if constraint_mask[i]:
        print(f"  {i:3d}: {name}")


# ============================================================
# EMPIRICAL TRAINING BOUNDS
# ============================================================

# Bounds are calculated ONLY from the training split.
#
# This prevents test-set information from determining the
# feasible perturbation region.

train_lower = np.min(X_train, axis=0)
train_upper = np.max(X_train, axis=0)

print("\nEmpirical bounds derived from training data.")


# ============================================================
# FIXED ATTACK-ELIGIBLE TEST SAMPLE SET
# ============================================================

model = joblib.load(MODEL_PATHS["mlp"])

clean_probability = (
    conventional_attacks.get_positive_class_probability(
        model,
        X_test
    )
)

clean_prediction = (
    clean_probability >= 0.5
).astype(int)

eligible_indices = np.where(
    (y_test == 1) &
    (clean_prediction == 1)
)[0]

rng = np.random.default_rng(SEED)

selected_indices = rng.choice(
    eligible_indices,
    size=N_SAMPLES,
    replace=False
)

selected_indices = np.sort(selected_indices)

X_attack = X_test[selected_indices].copy()
y_attack = y_test[selected_indices].copy()

print(
    f"\nFixed attack-eligible samples: "
    f"{len(selected_indices)}"
)


# ============================================================
# CONSTRAINT APPLICATION
# ============================================================

def apply_constraints(
    X_clean,
    X_candidate,
    feature_mask,
    lower_bounds,
    upper_bounds,
):
    """
    Apply the frozen constraint rules.

    1. Non-eligible features are restored to clean values.
    2. Eligible numerical features are clipped to empirical
       training-data bounds.
    """

    X_constrained = X_candidate.copy()

    # Restore every non-perturbable feature.
    X_constrained[:, ~feature_mask] = (
        X_clean[:, ~feature_mask]
    )

    # Enforce empirical training bounds only on eligible
    # numerical features.
    for i in np.where(feature_mask)[0]:
        X_constrained[:, i] = np.clip(
            X_constrained[:, i],
            lower_bounds[i],
            upper_bounds[i],
        )

    return X_constrained


# ============================================================
# CONSTRAINED FGSM
# ============================================================

def constrained_fgsm(
    model,
    X,
    y,
    epsilon,
    feature_mask,
    lower_bounds,
    upper_bounds,
):
    """
    FGSM followed by constraint projection.
    """

    X_adv = conventional_attacks.fgsm_attack_mlp(
        model=model,
        X=X,
        y=y,
        epsilon=epsilon,
        feature_mask=feature_mask,
    )

    X_adv = apply_constraints(
        X_clean=X,
        X_candidate=X_adv,
        feature_mask=feature_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return X_adv


# ============================================================
# CONSTRAINED PGD
# ============================================================

def constrained_pgd(
    model,
    X,
    y,
    epsilon,
    feature_mask,
    lower_bounds,
    upper_bounds,
):
    """
    PGD with both:
        - L-infinity perturbation budget
        - empirical training-data bounds
    """

    # Combine epsilon bounds with empirical feature bounds.
    lower = np.maximum(
        X - epsilon,
        lower_bounds
    )

    upper = np.minimum(
        X + epsilon,
        upper_bounds
    )

    step_size = epsilon / 4

    X_adv = conventional_attacks.pgd_attack_mlp(
        model=model,
        X=X,
        y=y,
        epsilon=epsilon,
        step_size=step_size,
        num_steps=PGD_STEPS,
        feature_mask=feature_mask,
        lower_bounds=lower,
        upper_bounds=upper,
    )

    X_adv = apply_constraints(
        X_clean=X,
        X_candidate=X_adv,
        feature_mask=feature_mask,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return X_adv


# ============================================================
# EVALUATION
# ============================================================

def evaluate_attack(model, X_clean, X_adv, y):
    """
    Calculate attack effectiveness and perturbation metrics.
    """

    clean_prob = (
        conventional_attacks
        .get_positive_class_probability(
            model,
            X_clean
        )
    )

    attacked_prob = (
        conventional_attacks
        .get_positive_class_probability(
            model,
            X_adv
        )
    )

    clean_pred = (clean_prob >= 0.5).astype(int)
    attacked_pred = (attacked_prob >= 0.5).astype(int)

    successful_evasion = (
        (y == 1) &
        (clean_pred == 1) &
        (attacked_pred == 0)
    )

    delta = X_adv - X_clean

    changed_features = np.sum(
        np.abs(delta) > 1e-12,
        axis=1
    )

    l2 = np.linalg.norm(
        delta,
        axis=1
    )

    linf = np.max(
        np.abs(delta),
        axis=1
    )

    # Verify non-eligible features were not changed.
    fixed_feature_change = np.max(
        np.abs(
            delta[:, ~constraint_mask]
        )
    )

    # Verify eligible features remain inside epsilon.
    max_perturbation = np.max(
        np.abs(delta[:, constraint_mask])
    )

    results = {
        "samples": len(y),
        "successful_evasions": int(
            successful_evasion.sum()
        ),
        "attack_success_rate": (
            successful_evasion.sum() / len(y)
        ),
        "mean_clean_malicious_probability": (
            np.mean(clean_prob)
        ),
        "mean_attacked_malicious_probability": (
            np.mean(attacked_prob)
        ),
        "mean_confidence_drop": (
            np.mean(clean_prob - attacked_prob)
        ),
        "mean_changed_features": (
            np.mean(changed_features)
        ),
        "mean_changed_feature_fraction": (
            np.mean(
                changed_features /
                len(feature_names)
            )
        ),
        "mean_l2": np.mean(l2),
        "mean_linf": np.mean(linf),
        "maximum_linf_observed": (
            np.max(linf)
        ),
        "fixed_feature_max_change": (
            fixed_feature_change
        ),
        "eligible_feature_max_change": (
            max_perturbation
        ),
    }

    return results


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []


for epsilon in EPSILONS:

    print("\n" + "=" * 70)
    print(
        f"CONSTRAINT-AWARE FGSM | "
        f"epsilon={epsilon}"
    )
    print("=" * 70)

    X_adv = constrained_fgsm(
        model=model,
        X=X_attack,
        y=y_attack,
        epsilon=epsilon,
        feature_mask=constraint_mask,
        lower_bounds=train_lower,
        upper_bounds=train_upper,
    )

    metrics = evaluate_attack(
        model,
        X_attack,
        X_adv,
        y_attack
    )

    metrics.update({
        "model": "mlp",
        "attack": "constraint_aware_FGSM",
        "epsilon": epsilon,
        "seed": SEED,
        "perturbable_features": int(
            constraint_mask.sum()
        ),
    })

    results.append(metrics)

    print(
        f"ASR: "
        f"{metrics['attack_success_rate']:.4f}"
    )

    print(
        f"Confidence drop: "
        f"{metrics['mean_confidence_drop']:.4f}"
    )

    print(
        f"Mean changed features: "
        f"{metrics['mean_changed_features']:.2f}"
    )

    print(
        f"Maximum L-inf observed: "
        f"{metrics['maximum_linf_observed']:.6f}"
    )

    print(
        f"Fixed-feature max change: "
        f"{metrics['fixed_feature_max_change']:.12f}"
    )


for epsilon in EPSILONS:

    print("\n" + "=" * 70)
    print(
        f"CONSTRAINT-AWARE PGD | "
        f"epsilon={epsilon}"
    )
    print("=" * 70)

    X_adv = constrained_pgd(
        model=model,
        X=X_attack,
        y=y_attack,
        epsilon=epsilon,
        feature_mask=constraint_mask,
        lower_bounds=train_lower,
        upper_bounds=train_upper,
    )

    metrics = evaluate_attack(
        model,
        X_attack,
        X_adv,
        y_attack
    )

    metrics.update({
        "model": "mlp",
        "attack": "constraint_aware_PGD",
        "epsilon": epsilon,
        "seed": SEED,
        "perturbable_features": int(
            constraint_mask.sum()
        ),
    })

    results.append(metrics)

    print(
        f"ASR: "
        f"{metrics['attack_success_rate']:.4f}"
    )

    print(
        f"Confidence drop: "
        f"{metrics['mean_confidence_drop']:.4f}"
    )

    print(
        f"Mean changed features: "
        f"{metrics['mean_changed_features']:.2f}"
    )

    print(
        f"Maximum L-inf observed: "
        f"{metrics['maximum_linf_observed']:.6f}"
    )

    print(
        f"Fixed-feature max change: "
        f"{metrics['fixed_feature_max_change']:.12f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR /
    "constraint_aware_attack_results_mlp.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


sample_path = (
    OUTPUT_DIR /
    "constraint_aware_attack_sample_indices.csv"
)

pd.DataFrame({
    "test_index": selected_indices
}).to_csv(
    sample_path,
    index=False
)


# ============================================================
# SAVE CONSTRAINT METADATA
# ============================================================

constraint_metadata = {
    "seed": SEED,
    "n_samples": N_SAMPLES,
    "epsilons": EPSILONS,
    "pgd_steps": PGD_STEPS,
    "total_processed_features": len(feature_names),
    "numerical_features": int(
        numerical_mask.sum()
    ),
    "constraint_eligible_features": int(
        constraint_mask.sum()
    ),
    "fixed_features": int(
        (~constraint_mask).sum()
    ),
    "constraint_type": (
        "constraint-aware feature-space"
    ),
    "host_space_attack": False,
    "bounds_source": (
        "UNSW-NB15 training split only"
    ),
    "categorical_features_fixed": True,
    "binary_semantic_features_fixed": True,
    "contextual_derived_features_fixed": True,
}

metadata_output = (
    OUTPUT_DIR /
    "constraint_aware_attack_metadata_mlp.json"
)

with open(
    metadata_output,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        constraint_metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CONSTRAINT-AWARE EXPERIMENT COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "model",
            "attack",
            "epsilon",
            "samples",
            "successful_evasions",
            "attack_success_rate",
            "mean_confidence_drop",
            "mean_changed_features",
            "mean_l2",
            "mean_linf",
            "fixed_feature_max_change",
        ]
    ].to_string(index=False)
)

print("\nResults:")
print(results_path)

print("\nSample indices:")
print(sample_path)

print("\nMetadata:")
print(metadata_output)