"""
Effect-size and bootstrap analysis for conventional vs
constraint-aware adversarial attacks.

Comparisons:
    Conventional FGSM vs Constraint-aware FGSM
    Conventional PGD  vs Constraint-aware PGD

Budgets:
    epsilon = 0.05, 0.10, 0.20

Outputs:
    - paired confidence-drop summaries
    - bootstrap 95% CI for confidence-drop difference
    - bootstrap 95% CI for ASR difference
    - probability that conventional confidence drop > constrained
    - paired evasion contingency counts

Important:
    Bootstrap resamples the SAME paired test instances.
    This preserves the paired experimental design.
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

N_BOOTSTRAP = 10000

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
print("EFFECT SIZE AND BOOTSTRAP ANALYSIS")
print("=" * 70)

X_test = pd.read_csv(X_TEST_PATH).values
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

X_train = pd.read_csv(X_TRAIN_PATH).values

model = joblib.load(MODEL_PATH)

sample_indices = pd.read_csv(
    SAMPLE_PATH
)["test_index"].values

X_attack = X_test[sample_indices].copy()
y_attack = y_test[sample_indices].copy()

print(f"Analysis samples: {len(X_attack)}")
print(f"Bootstrap iterations: {N_BOOTSTRAP}")


# ============================================================
# FEATURE MASKS
# ============================================================

conventional_mask = (
    conventional_attacks
    .get_numerical_feature_mask(
        n_features=X_test.shape[1]
    )
)


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

encoder = preprocessor.named_transformers_[
    "categorical"
]

encoded_feature_names = (
    encoder.get_feature_names_out(
        metadata["categorical_features"]
    )
)

feature_names.extend(
    f"categorical__{name}"
    for name in encoded_feature_names
)


constraint_mask = conventional_mask.copy()

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


print(
    f"Conventional perturbable features: "
    f"{conventional_mask.sum()}"
)

print(
    f"Constraint-aware perturbable features: "
    f"{constraint_mask.sum()}"
)


# ============================================================
# TRAINING BOUNDS
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
# PROBABILITY
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
# CONSTRAINT APPLICATION
# ============================================================

def apply_constraints(
    X_clean,
    X_candidate,
):

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
# ATTACK GENERATION
# ============================================================

def generate_attack(
    attack_type,
    epsilon,
):

    if attack_type == "FGSM":

        return conventional_attacks.fgsm_attack_mlp(
            model=model,
            X=X_attack,
            y=y_attack,
            epsilon=epsilon,
            feature_mask=conventional_mask,
        )


    if attack_type == "PGD":

        return conventional_attacks.pgd_attack_mlp(
            model=model,
            X=X_attack,
            y=y_attack,
            epsilon=epsilon,
            step_size=epsilon / 4,
            num_steps=PGD_STEPS,
            feature_mask=conventional_mask,
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
# CLEAN PROBABILITY
# ============================================================

clean_probability = get_probability(
    X_attack
)

clean_prediction = (
    clean_probability >= 0.5
).astype(int)


# ============================================================
# BOOTSTRAP FUNCTION
# ============================================================

def bootstrap_mean_difference(
    differences,
    rng,
    n_bootstrap=10000,
):

    n = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap
    )

    for i in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        bootstrap_means[i] = np.mean(
            differences[indices]
        )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    return (
        np.mean(differences),
        lower,
        upper,
    )


# ============================================================
# ANALYSIS
# ============================================================

rng = np.random.default_rng(SEED)

results = []


for attack_family in ["FGSM", "PGD"]:

    for epsilon in EPSILONS:

        print("\n" + "=" * 70)
        print(
            f"{attack_family} | epsilon={epsilon}"
        )
        print("=" * 70)


        # ----------------------------------------------------
        # Generate paired attacks
        # ----------------------------------------------------

        X_conv = generate_attack(
            attack_family,
            epsilon
        )

        X_const = generate_attack(
            f"constraint_aware_{attack_family}",
            epsilon
        )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        p_conv = get_probability(
            X_conv
        )

        p_const = get_probability(
            X_const
        )


        # ----------------------------------------------------
        # Confidence drops
        # ----------------------------------------------------

        confidence_drop_conv = (
            clean_probability -
            p_conv
        )

        confidence_drop_const = (
            clean_probability -
            p_const
        )

        confidence_difference = (
            confidence_drop_conv -
            confidence_drop_const
        )


        # ----------------------------------------------------
        # ASR indicators
        # ----------------------------------------------------

        pred_conv = (
            p_conv >= 0.5
        ).astype(int)

        pred_const = (
            p_const >= 0.5
        ).astype(int)


        conv_success = (
            (y_attack == 1) &
            (clean_prediction == 1) &
            (pred_conv == 0)
        )

        const_success = (
            (y_attack == 1) &
            (clean_prediction == 1) &
            (pred_const == 0)
        )


        # ----------------------------------------------------
        # Bootstrap confidence difference
        # ----------------------------------------------------

        (
            mean_conf_diff,
            conf_ci_low,
            conf_ci_high,
        ) = bootstrap_mean_difference(
            confidence_difference,
            rng,
            N_BOOTSTRAP
        )


        # ----------------------------------------------------
        # Bootstrap ASR difference
        # ----------------------------------------------------

        asr_difference_per_sample = (
            conv_success.astype(float) -
            const_success.astype(float)
        )

        (
            mean_asr_diff,
            asr_ci_low,
            asr_ci_high,
        ) = bootstrap_mean_difference(
            asr_difference_per_sample,
            rng,
            N_BOOTSTRAP
        )


        # ----------------------------------------------------
        # Probability conventional confidence drop is larger
        # ----------------------------------------------------

        conventional_greater = np.mean(
            confidence_difference > 0
        )


        # ----------------------------------------------------
        # Median and IQR
        # ----------------------------------------------------

        median_difference = np.median(
            confidence_difference
        )

        q1 = np.percentile(
            confidence_difference,
            25
        )

        q3 = np.percentile(
            confidence_difference,
            75
        )


        # ----------------------------------------------------
        # Paired evasion counts
        # ----------------------------------------------------

        both_fail = np.sum(
            (~conv_success) &
            (~const_success)
        )

        conventional_only = np.sum(
            conv_success &
            (~const_success)
        )

        constraint_only = np.sum(
            (~conv_success) &
            const_success
        )

        both_success = np.sum(
            conv_success &
            const_success
        )


        # ----------------------------------------------------
        # Relative ASR reduction
        # ----------------------------------------------------

        conv_asr = np.mean(
            conv_success
        )

        const_asr = np.mean(
            const_success
        )

        if conv_asr > 0:

            relative_reduction = (
                1 -
                const_asr /
                conv_asr
            )

        else:

            relative_reduction = np.nan


        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        result = {

            "model": "mlp",

            "attack": attack_family,

            "epsilon": epsilon,

            "samples": len(y_attack),

            "conventional_asr": conv_asr,

            "constraint_aware_asr": const_asr,

            "asr_difference": mean_asr_diff,

            "asr_bootstrap_ci_lower": (
                asr_ci_low
            ),

            "asr_bootstrap_ci_upper": (
                asr_ci_high
            ),

            "relative_asr_reduction": (
                relative_reduction
            ),

            "conventional_mean_confidence_drop": (
                np.mean(confidence_drop_conv)
            ),

            "constraint_aware_mean_confidence_drop": (
                np.mean(confidence_drop_const)
            ),

            "confidence_drop_difference": (
                mean_conf_diff
            ),

            "confidence_bootstrap_ci_lower": (
                conf_ci_low
            ),

            "confidence_bootstrap_ci_upper": (
                conf_ci_high
            ),

            "median_confidence_difference": (
                median_difference
            ),

            "confidence_difference_q1": q1,

            "confidence_difference_q3": q3,

            "conventional_greater_confidence_fraction": (
                conventional_greater
            ),

            "both_fail": int(
                both_fail
            ),

            "conventional_only": int(
                conventional_only
            ),

            "constraint_only": int(
                constraint_only
            ),

            "both_success": int(
                both_success
            ),
        }

        results.append(result)


        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"Conventional ASR: "
            f"{conv_asr:.4f}"
        )

        print(
            f"Constraint-aware ASR: "
            f"{const_asr:.4f}"
        )

        print(
            f"ASR difference: "
            f"{mean_asr_diff:.4f} "
            f"(95% CI "
            f"{asr_ci_low:.4f}, "
            f"{asr_ci_high:.4f})"
        )

        print(
            f"Confidence-drop difference: "
            f"{mean_conf_diff:.4f} "
            f"(95% CI "
            f"{conf_ci_low:.4f}, "
            f"{conf_ci_high:.4f})"
        )

        print(
            f"Median confidence difference: "
            f"{median_difference:.4f}"
        )

        print(
            f"Conventional greater confidence drop: "
            f"{conventional_greater:.4f}"
        )

        print(
            f"Paired evasion counts:"
        )

        print(
            f"  Both fail:          {both_fail}"
        )

        print(
            f"  Conventional only:  {conventional_only}"
        )

        print(
            f"  Constraint only:    {constraint_only}"
        )

        print(
            f"  Both succeed:       {both_success}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_path = (
    OUTPUT_DIR /
    "effect_size_bootstrap_results_mlp.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_output = (
    OUTPUT_DIR /
    "effect_size_bootstrap_metadata_mlp.json"
)

analysis_metadata = {

    "model": "mlp",

    "dataset": "UNSW-NB15",

    "seed": SEED,

    "samples": len(X_attack),

    "bootstrap_iterations": N_BOOTSTRAP,

    "bootstrap_confidence_level": 0.95,

    "resampling_unit": "paired test instance",

    "comparison": (
        "conventional vs constraint-aware"
    ),

    "attacks": [
        "FGSM",
        "PGD"
    ],

    "epsilons": EPSILONS,

    "constraint_type": (
        "constraint-aware feature-space"
    ),

    "host_space_attack": False,
}

with open(
    metadata_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        analysis_metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("EFFECT SIZE AND BOOTSTRAP ANALYSIS COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "attack",
            "epsilon",
            "conventional_asr",
            "constraint_aware_asr",
            "asr_difference",
            "asr_bootstrap_ci_lower",
            "asr_bootstrap_ci_upper",
            "relative_asr_reduction",
            "confidence_drop_difference",
            "confidence_bootstrap_ci_lower",
            "confidence_bootstrap_ci_upper",
            "median_confidence_difference",
            "conventional_greater_confidence_fraction",
        ]
    ].to_string(index=False)
)

print("\nResults:")
print(output_path)

print("\nMetadata:")
print(metadata_output)