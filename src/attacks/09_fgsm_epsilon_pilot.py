from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

# Allow imports from this directory.
ATTACK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ATTACK_DIR))

import importlib.util

ATTACK_MODULE = ATTACK_DIR / "06_conventional_attacks.py"

spec = importlib.util.spec_from_file_location(
    "conventional_attacks",
    ATTACK_MODULE,
)

conventional_attacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conventional_attacks)

PROJECT_ROOT = ATTACK_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "attacks"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_test = pd.read_csv(DATA_DIR / "unsw_X_test.csv").values
    y_test = pd.read_csv(DATA_DIR / "unsw_y_test.csv").values.ravel()

    return X_test, y_test


def main():

    print("FGSM epsilon pilot")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load model and test data
    # ---------------------------------------------------------

    model = joblib.load(
        MODEL_DIR / "mlp_clean.joblib"
    )

    X_test, y_test = load_data()

    # ---------------------------------------------------------
    # Clean predictions
    # ---------------------------------------------------------

    clean_pred = model.predict(X_test)

    eligible_indices = conventional_attacks.get_attack_eligible_indices(
        y_test,
        clean_pred,
    )

    print(f"Total test samples: {len(X_test)}")
    print(f"Attack-eligible samples: {len(eligible_indices)}")

    # Use a fixed subset for the pilot so every epsilon is
    # evaluated on exactly the same observations.
    pilot_size = min(1000, len(eligible_indices))

    rng = np.random.default_rng(42)

    selected_indices = rng.choice(
        eligible_indices,
        size=pilot_size,
        replace=False,
    )

    X_clean = X_test[selected_indices]
    y_clean = y_test[selected_indices]

    # ---------------------------------------------------------
    # Attack mask
    # ---------------------------------------------------------
    #
    # First 39 processed features are numerical.
    # Remaining 153 features are one-hot categorical.
    #
    # Categorical representation remains unchanged.
    # ---------------------------------------------------------

    feature_mask = conventional_attacks.get_numerical_feature_mask(
        X_clean.shape[1]
    )

    print(f"Perturbable numerical features: {feature_mask.sum()}")
    print(
        f"Fixed categorical features: "
        f"{(~feature_mask).sum()}"
    )

    # ---------------------------------------------------------
    # Epsilon sweep
    # ---------------------------------------------------------

    epsilons = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.50,
    ]

    results = []

    for epsilon in epsilons:

        print()
        print(f"Running epsilon = {epsilon}")

        X_adv = conventional_attacks.fgsm_attack_mlp(
            model=model,
            X=X_clean,
            y=y_clean,
            epsilon=epsilon,
            feature_mask=feature_mask,
        )

        metrics = conventional_attacks.evaluate_adversarial_examples(
            model=model,
            X_clean=X_clean,
            X_adv=X_adv,
            y_true=y_clean,
        )

        metrics["model"] = "mlp"
        metrics["attack"] = "fgsm"
        metrics["epsilon"] = epsilon
        metrics["pilot_samples"] = pilot_size
        metrics["seed"] = 42

        results.append(metrics)

        print(
            f"ASR: {metrics['attack_success_rate']:.4f} | "
            f"Confidence drop: "
            f"{metrics['mean_confidence_drop']:.4f} | "
            f"Changed features: "
            f"{metrics['mean_changed_features']:.1f} | "
            f"L2: {metrics['mean_l2']:.4f} | "
            f"Linf: {metrics['mean_linf']:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_file = (
        RESULTS_DIR /
        "fgsm_epsilon_pilot_mlp.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("Pilot complete.")
    print(f"Saved: {output_file}")
    print()
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()