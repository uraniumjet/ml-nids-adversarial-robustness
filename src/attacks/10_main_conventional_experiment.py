"""
Main conventional feature-space adversarial attack experiment.

Models:
    - MLP: gradient-based FGSM and PGD
    - Random Forest: query-based black-box coordinate attack
    - XGBoost: query-based black-box coordinate attack

Budgets:
    epsilon = 0.05, 0.10, 0.20

Design:
    - Fixed 1,000 attack-eligible test samples
    - Only the 39 numerical features are perturbable
    - Categorical one-hot features remain unchanged
    - No model retraining
    - Results saved to results/attacks/
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

X_TEST_PATH = ROOT / "data/processed/unsw_X_test.csv"
Y_TEST_PATH = ROOT / "data/processed/unsw_y_test.csv"

MODEL_PATHS = {
    "mlp": ROOT / "results/models/mlp_clean.joblib",
    "randomforest": ROOT / "results/models/randomforest_clean.joblib",
    "xgboost": ROOT / "results/models/xgboost_clean.joblib",
}

OUTPUT_DIR = ROOT / "results/attacks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONVENTIONAL_MODULE_PATH = ROOT / "src/attacks/06_conventional_attacks.py"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42
N_SAMPLES = 1000

EPSILONS = [0.05, 0.10, 0.20]

# PGD settings
PGD_STEPS = 10

# For black-box attacks:
# Each iteration evaluates +step and -step for candidate features.
BLACK_BOX_STEPS = 10
BLACK_BOX_FEATURES_PER_STEP = 10

# Use all 39 numerical features as candidates.
N_NUMERICAL_FEATURES = 39


# ============================================================
# LOAD EXISTING ATTACK MODULE
# ============================================================

spec = importlib.util.spec_from_file_location(
    "conventional_attacks",
    CONVENTIONAL_MODULE_PATH
)

conventional_attacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conventional_attacks)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MAIN CONVENTIONAL ATTACK EXPERIMENT")
print("=" * 70)

X_test = pd.read_csv(X_TEST_PATH).values
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print(f"Test samples: {X_test.shape[0]}")
print(f"Processed features: {X_test.shape[1]}")


# ============================================================
# NUMERICAL FEATURE MASK
# ============================================================

feature_mask = conventional_attacks.get_numerical_feature_mask(
    n_features=X_test.shape[1]
)

print(f"Perturbable numerical features: {feature_mask.sum()}")
print(f"Fixed categorical features: {(~feature_mask).sum()}")


# ============================================================
# FIXED ATTACK-ELIGIBLE SAMPLE SET
# ============================================================

sample_rng = np.random.default_rng(SEED)

eligible_all = []

# Use the MLP only to establish clean-correct malicious samples.
# The same fixed sample IDs are then used for all models.
reference_model = joblib.load(MODEL_PATHS["mlp"])

reference_prob = conventional_attacks.get_positive_class_probability(
    reference_model,
    X_test
)

reference_pred = (reference_prob >= 0.5).astype(int)

eligible_all = np.where(
    (y_test == 1) &
    (reference_pred == 1)
)[0]

if len(eligible_all) < N_SAMPLES:
    raise RuntimeError(
        f"Only {len(eligible_all)} eligible samples available; "
        f"need {N_SAMPLES}."
    )

selected_indices = sample_rng.choice(
    eligible_all,
    size=N_SAMPLES,
    replace=False
)

selected_indices = np.sort(selected_indices)

X_attack = X_test[selected_indices].copy()
y_attack = y_test[selected_indices].copy()

print(f"Attack-eligible samples selected: {len(selected_indices)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_probability(model, X):
    """
    Return probability of the malicious/positive class.
    """
    return conventional_attacks.get_positive_class_probability(
        model,
        X
    )


def calculate_metrics(model, X_clean, X_adv, y):
    """
    Calculate adversarial evaluation metrics.
    """

    clean_prob = get_probability(model, X_clean)
    adv_prob = get_probability(model, X_adv)

    clean_pred = (clean_prob >= 0.5).astype(int)
    adv_pred = (adv_prob >= 0.5).astype(int)

    successful = (
        (y == 1) &
        (clean_pred == 1) &
        (adv_pred == 0)
    )

    asr = successful.sum() / len(y)

    clean_accuracy = np.mean(clean_pred == y)
    attacked_accuracy = np.mean(adv_pred == y)

    tp = np.sum((y == 1) & (adv_pred == 1))
    fn = np.sum((y == 1) & (adv_pred == 0))
    fp = np.sum((y == 0) & (adv_pred == 1))
    tn = np.sum((y == 0) & (adv_pred == 0))

    # Attack set is malicious only, so FPR/FNR here are mainly
    # supplementary diagnostics. FNR on the attack set is directly
    # related to evasion.
    total_negative = fp + tn
    total_positive = tp + fn

    fpr = fp / total_negative if total_negative > 0 else np.nan
    fnr = fn / total_positive if total_positive > 0 else np.nan

    delta = X_adv - X_clean

    changed_fraction = np.mean(
        np.abs(delta) > 1e-12,
        axis=1
    )

    l2 = np.linalg.norm(delta, axis=1)
    linf = np.max(np.abs(delta), axis=1)

    confidence_drop = clean_prob - adv_prob

    return {
        "attack_success_rate": asr,
        "clean_accuracy": clean_accuracy,
        "attacked_accuracy": attacked_accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "mean_changed_feature_fraction": np.mean(changed_fraction),
        "mean_changed_features": np.mean(
            np.sum(np.abs(delta) > 1e-12, axis=1)
        ),
        "mean_l2": np.mean(l2),
        "mean_linf": np.mean(linf),
        "mean_clean_malicious_probability": np.mean(clean_prob),
        "mean_attacked_malicious_probability": np.mean(adv_prob),
        "mean_confidence_drop": np.mean(confidence_drop),
        "successful_evasions": int(successful.sum()),
    }


def black_box_coordinate_attack(
    model,
    X,
    epsilon,
    feature_mask,
    steps=10,
    features_per_step=10,
):
    """
    Query-based black-box coordinate attack.

    At each iteration:
      1. Select candidate numerical features.
      2. Test movement in the positive and negative directions.
      3. Keep the perturbation producing the largest reduction
         in malicious-class probability.
      4. Respect the L-infinity epsilon budget.

    This is deliberately model-agnostic and uses only model queries.
    """

    X_adv = X.copy()

    feature_indices = np.where(feature_mask)[0]

    # Deterministic feature ordering.
    # We use all numerical dimensions but limit the number considered
    # per iteration to keep query cost manageable.
    feature_indices = feature_indices[:features_per_step]

    step_size = epsilon / steps

    for iteration in range(steps):

        for feature_idx in feature_indices:

            current = X_adv[:, feature_idx].copy()

            plus = X_adv.copy()
            minus = X_adv.copy()

            plus[:, feature_idx] = current + step_size
            minus[:, feature_idx] = current - step_size

            # Enforce the global L-infinity budget relative to clean X.
            plus[:, feature_idx] = np.clip(
                plus[:, feature_idx],
                X[:, feature_idx] - epsilon,
                X[:, feature_idx] + epsilon
            )

            minus[:, feature_idx] = np.clip(
                minus[:, feature_idx],
                X[:, feature_idx] - epsilon,
                X[:, feature_idx] + epsilon
            )

            current_prob = get_probability(model, X_adv)
            plus_prob = get_probability(model, plus)
            minus_prob = get_probability(model, minus)

            best_prob = current_prob.copy()

            use_plus = plus_prob < best_prob
            use_minus = (
                (minus_prob < best_prob) &
                (minus_prob <= plus_prob)
            )

            X_adv[use_plus, feature_idx] = plus[
                use_plus,
                feature_idx
            ]

            X_adv[use_minus, feature_idx] = minus[
                use_minus,
                feature_idx
            ]

        print(
            f"    black-box iteration "
            f"{iteration + 1}/{steps}"
        )

    return X_adv


# ============================================================
# MAIN EXPERIMENT
# ============================================================

results = []

for model_name, model_path in MODEL_PATHS.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    model = joblib.load(model_path)

    clean_prob = get_probability(model, X_attack)
    clean_pred = (clean_prob >= 0.5).astype(int)

    # Keep only samples correctly classified as malicious
    # by the current model.
    model_eligible = (
        (y_attack == 1) &
        (clean_pred == 1)
    )

    X_model = X_attack[model_eligible].copy()
    y_model = y_attack[model_eligible].copy()

    print(
        f"Model-specific attack-eligible samples: "
        f"{len(X_model)}"
    )

    if len(X_model) == 0:
        print("No eligible samples. Skipping model.")
        continue

    # --------------------------------------------------------
    # FGSM / PGD for MLP
    # --------------------------------------------------------

    if model_name == "mlp":

        for epsilon in EPSILONS:

            print(f"\nMLP FGSM | epsilon={epsilon}")

            X_adv = conventional_attacks.fgsm_attack_mlp(
                model=model,
                X=X_model,
                y=y_model,
                epsilon=epsilon,
                feature_mask=feature_mask,
            )

            metrics = calculate_metrics(
                model,
                X_model,
                X_adv,
                y_model
            )

            metrics.update({
                "model": model_name,
                "attack": "FGSM",
                "epsilon": epsilon,
                "samples": len(X_model),
                "seed": SEED,
            })

            results.append(metrics)

            print(
                f"ASR: {metrics['attack_success_rate']:.4f} | "
                f"confidence drop: "
                f"{metrics['mean_confidence_drop']:.4f}"
            )

            print(f"\nMLP PGD | epsilon={epsilon}")

            alpha = epsilon / 4

            X_adv = conventional_attacks.pgd_attack_mlp(
                model=model,
                X=X_model,
                y=y_model,
                epsilon=epsilon,
                step_size=alpha,
                num_steps=PGD_STEPS,
                feature_mask=feature_mask,
            )

            metrics = calculate_metrics(
                model,
                X_model,
                X_adv,
                y_model
            )

            metrics.update({
                "model": model_name,
                "attack": "PGD",
                "epsilon": epsilon,
                "samples": len(X_model),
                "seed": SEED,
            })

            results.append(metrics)

            print(
                f"ASR: {metrics['attack_success_rate']:.4f} | "
                f"confidence drop: "
                f"{metrics['mean_confidence_drop']:.4f}"
            )

    # --------------------------------------------------------
    # BLACK-BOX ATTACK FOR RF / XGBOOST
    # --------------------------------------------------------

    else:

        for epsilon in EPSILONS:

            print(
                f"\n{model_name} BLACK-BOX "
                f"| epsilon={epsilon}"
            )

            X_adv = black_box_coordinate_attack(
                model=model,
                X=X_model,
                epsilon=epsilon,
                feature_mask=feature_mask,
                steps=BLACK_BOX_STEPS,
                features_per_step=BLACK_BOX_FEATURES_PER_STEP,
            )

            metrics = calculate_metrics(
                model,
                X_model,
                X_adv,
                y_model
            )

            metrics.update({
                "model": model_name,
                "attack": "black_box_coordinate",
                "epsilon": epsilon,
                "samples": len(X_model),
                "seed": SEED,
                "black_box_steps": BLACK_BOX_STEPS,
                "black_box_features_per_step":
                    BLACK_BOX_FEATURES_PER_STEP,
            })

            results.append(metrics)

            print(
                f"ASR: {metrics['attack_success_rate']:.4f} | "
                f"confidence drop: "
                f"{metrics['mean_confidence_drop']:.4f}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_path = (
    OUTPUT_DIR /
    "conventional_attack_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

# Save exact selected sample IDs for reproducibility.
sample_path = (
    OUTPUT_DIR /
    "conventional_attack_sample_indices.csv"
)

pd.DataFrame({
    "test_index": selected_indices
}).to_csv(
    sample_path,
    index=False
)


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print(results_df[
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
    ]
].to_string(index=False))

print("\nResults saved to:")
print(output_path)

print("\nSample indices saved to:")
print(sample_path)