"""
Statistical analysis of conventional vs constraint-aware attacks.

Primary comparison:
    Conventional attack vs constraint-aware attack

At each epsilon:
    FGSM conventional vs constraint-aware FGSM
    PGD conventional vs constraint-aware PGD

Metrics:
    - confidence drop
    - absolute probability change
    - evasion indicator

Statistical tests:
    - Wilcoxon signed-rank test for paired continuous outcomes
    - McNemar's test for paired binary evasion outcomes
    - Effect sizes

All attacks are regenerated using the same:
    - model
    - test samples
    - random seed
    - attack budgets
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import joblib

from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


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
print("STATISTICAL ANALYSIS")
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
print(f"Processed features: {X_attack.shape[1]}")


# ============================================================
# FEATURE MASKS
# ============================================================

conventional_mask = (
    conventional_attacks
    .get_numerical_feature_mask(
        n_features=X_test.shape[1]
    )
)


# Reconstruct processed feature names
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
# TRAINING-DATA BOUNDS
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
# PROBABILITY FUNCTION
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
# CONSTRAINT FUNCTION
# ============================================================

def apply_constraints(X_clean, X_candidate):

    X_constrained = X_candidate.copy()

    # Restore all fixed features.
    X_constrained[:, ~constraint_mask] = (
        X_clean[:, ~constraint_mask]
    )

    # Enforce empirical training bounds.
    for i in np.where(constraint_mask)[0]:

        X_constrained[:, i] = np.clip(
            X_constrained[:, i],
            train_lower[i],
            train_upper[i],
        )

    return X_constrained


# ============================================================
# ATTACK GENERATORS
# ============================================================

def generate_attack(
    attack_type,
    epsilon,
):
    """
    Generate exactly one attack condition.
    """

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
# EFFECT SIZE
# ============================================================

def rank_biserial_effect(
    x,
    y,
):
    """
    Rank-biserial correlation for paired data.

    Positive values mean x tends to be larger than y.
    """

    differences = np.asarray(x) - np.asarray(y)

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    abs_diff = np.abs(differences)

    ranks = pd.Series(abs_diff).rank(
        method="average"
    ).values

    positive_rank_sum = np.sum(
        ranks[differences > 0]
    )

    negative_rank_sum = np.sum(
        ranks[differences < 0]
    )

    denominator = (
        positive_rank_sum +
        negative_rank_sum
    )

    if denominator == 0:
        return 0.0

    return (
        positive_rank_sum -
        negative_rank_sum
    ) / denominator


# ============================================================
# MCNEMAR EFFECT
# ============================================================

def mcnemar_table(
    conventional_success,
    constrained_success,
):
    """
    Build the paired 2x2 table.

    Rows:
        conventional

    Columns:
        constraint-aware
    """

    both_fail = np.sum(
        (~conventional_success) &
        (~constrained_success)
    )

    conventional_only = np.sum(
        conventional_success &
        (~constrained_success)
    )

    constrained_only = np.sum(
        (~conventional_success) &
        constrained_success
    )

    both_success = np.sum(
        conventional_success &
        constrained_success
    )

    table = np.array([
        [both_fail, conventional_only],
        [constrained_only, both_success],
    ])

    return table


# ============================================================
# CLEAN BASELINE
# ============================================================

clean_probability = get_probability(
    X_attack
)

clean_prediction = (
    clean_probability >= 0.5
).astype(int)

print(
    f"Clean malicious predictions: "
    f"{clean_prediction.sum()}"
)


# ============================================================
# RUN PAIRED COMPARISONS
# ============================================================

results = []


for attack_family in ["FGSM", "PGD"]:

    for epsilon in EPSILONS:

        print("\n" + "=" * 70)
        print(
            f"{attack_family} | "
            f"epsilon={epsilon}"
        )
        print("=" * 70)

        conventional_type = attack_family

        constrained_type = (
            f"constraint_aware_{attack_family}"
        )

        # Generate paired attacks.
        X_conv = generate_attack(
            conventional_type,
            epsilon
        )

        X_const = generate_attack(
            constrained_type,
            epsilon
        )

        p_conv = get_probability(
            X_conv
        )

        p_const = get_probability(
            X_const
        )

        pred_conv = (
            p_conv >= 0.5
        ).astype(int)

        pred_const = (
            p_const >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Confidence changes relative to clean condition
        # ----------------------------------------------------

        clean_to_conv = (
            clean_probability -
            p_conv
        )

        clean_to_const = (
            clean_probability -
            p_const
        )

        # ----------------------------------------------------
        # Wilcoxon: confidence drop
        # ----------------------------------------------------

        try:
            wilcox = wilcoxon(
                clean_to_conv,
                clean_to_const,
                alternative="two-sided",
                zero_method="wilcox",
            )

            wilcoxon_statistic = (
                float(wilcox.statistic)
            )

            wilcoxon_p = (
                float(wilcox.pvalue)
            )

        except ValueError:

            wilcoxon_statistic = np.nan
            wilcoxon_p = np.nan

        effect_size = rank_biserial_effect(
            clean_to_conv,
            clean_to_const
        )

        # ----------------------------------------------------
        # Evasion indicators
        # ----------------------------------------------------

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
        # McNemar
        # ----------------------------------------------------

        table = mcnemar_table(
            conv_success,
            const_success
        )

        exact_result = mcnemar(
            table,
            exact=True
        )

        mcnemar_statistic = (
            float(exact_result.statistic)
        )

        mcnemar_p = (
            float(exact_result.pvalue)
        )

        # ----------------------------------------------------
        # Differences
        # ----------------------------------------------------

        mean_confidence_difference = (
            np.mean(clean_to_conv) -
            np.mean(clean_to_const)
        )

        asr_difference = (
            np.mean(conv_success) -
            np.mean(const_success)
        )

        relative_asr_reduction = np.nan

        if np.mean(conv_success) > 0:

            relative_asr_reduction = (
                1 -
                (
                    np.mean(const_success) /
                    np.mean(conv_success)
                )
            )

        result = {
            "model": "mlp",
            "attack": attack_family,
            "epsilon": epsilon,

            "samples": len(y_attack),

            "conventional_asr": (
                np.mean(conv_success)
            ),

            "constraint_aware_asr": (
                np.mean(const_success)
            ),

            "asr_difference": (
                asr_difference
            ),

            "relative_asr_reduction": (
                relative_asr_reduction
            ),

            "conventional_mean_confidence_drop": (
                np.mean(clean_to_conv)
            ),

            "constraint_aware_mean_confidence_drop": (
                np.mean(clean_to_const)
            ),

            "confidence_drop_difference": (
                mean_confidence_difference
            ),

            "wilcoxon_statistic": (
                wilcoxon_statistic
            ),

            "wilcoxon_p_value": (
                wilcoxon_p
            ),

            "rank_biserial_effect": (
                effect_size
            ),

            "mcnemar_statistic": (
                mcnemar_statistic
            ),

            "mcnemar_p_value": (
                mcnemar_p
            ),

            "both_fail": int(table[0, 0]),
            "conventional_only": int(table[0, 1]),
            "constraint_only": int(table[1, 0]),
            "both_success": int(table[1, 1]),
        }

        results.append(result)

        print(
            f"Conventional ASR: "
            f"{result['conventional_asr']:.4f}"
        )

        print(
            f"Constraint-aware ASR: "
            f"{result['constraint_aware_asr']:.4f}"
        )

        print(
            f"ASR difference: "
            f"{result['asr_difference']:.4f}"
        )

        print(
            f"Relative ASR reduction: "
            f"{result['relative_asr_reduction']:.4f}"
        )

        print(
            f"Confidence-drop difference: "
            f"{result['confidence_drop_difference']:.4f}"
        )

        print(
            f"Wilcoxon p-value: "
            f"{result['wilcoxon_p_value']:.6g}"
        )

        print(
            f"McNemar p-value: "
            f"{result['mcnemar_p_value']:.6g}"
        )

        print(
            f"Rank-biserial effect: "
            f"{result['rank_biserial_effect']:.4f}"
        )


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

results_df = pd.DataFrame(results)

# Holm correction for the six Wilcoxon comparisons
# and separately for the six McNemar comparisons.

def holm_adjust(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    order = np.argsort(p_values)

    adjusted = np.empty_like(
        p_values
    )

    m = len(p_values)

    previous = 0.0

    for rank, index in enumerate(order):

        value = (
            (m - rank) *
            p_values[index]
        )

        value = max(
            value,
            previous
        )

        adjusted[index] = min(
            value,
            1.0
        )

        previous = adjusted[index]

    return adjusted


results_df["wilcoxon_p_holm"] = (
    holm_adjust(
        results_df["wilcoxon_p_value"].values
    )
)

results_df["mcnemar_p_holm"] = (
    holm_adjust(
        results_df["mcnemar_p_value"].values
    )
)


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR /
    "statistical_comparison_results_mlp.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "attack",
            "epsilon",
            "conventional_asr",
            "constraint_aware_asr",
            "asr_difference",
            "relative_asr_reduction",
            "confidence_drop_difference",
            "wilcoxon_p_value",
            "wilcoxon_p_holm",
            "mcnemar_p_value",
            "mcnemar_p_holm",
            "rank_biserial_effect",
        ]
    ].to_string(index=False)
)

print("\nResults:")
print(output_path)