from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ATTACK_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "attacks"
    / "06_conventional_attacks.py"
)

spec = importlib.util.spec_from_file_location(
    "conventional_attacks",
    ATTACK_MODULE_PATH,
)

conventional_attacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conventional_attacks)


def load_unsw_test_data():
    """
    Load the processed UNSW-NB15 test set.
    """

    X_test = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "unsw_X_test.csv"
    )

    y_test = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "unsw_y_test.csv"
    ).squeeze("columns")

    return (
        X_test.to_numpy(dtype=float),
        y_test.to_numpy(),
    )


def load_model(model_name: str):
    """
    Load one of the frozen clean baseline models.
    """

    model_paths = {
        "mlp": PROJECT_ROOT / "results" / "models" / "mlp_clean.joblib",
        "randomforest": PROJECT_ROOT / "results" / "models" / "randomforest_clean.joblib",
        "xgboost": PROJECT_ROOT / "results" / "models" / "xgboost_clean.joblib",
    }

    if model_name not in model_paths:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(model_paths)}"
        )

    return joblib.load(model_paths[model_name])


def get_clean_predictions(model, X):
    """
    Generate clean predictions and malicious-class probabilities.
    """

    predictions = model.predict(X)
    probabilities = conventional_attacks.get_positive_class_probability(
        model,
        X,
    )

    return predictions, probabilities


def select_attack_eligible_samples(
    y_true,
    clean_predictions,
):
    """
    Select malicious observations correctly detected by the
    clean model.

    These observations form the attack-eligible population.
    """

    return conventional_attacks.get_attack_eligible_indices(
        y_true,
        clean_predictions,
    )

def evaluate_adversarial_examples(
    y_true,
    clean_predictions,
    clean_probabilities,
    attacked_predictions,
    attacked_probabilities,
    X_original,
    X_attacked,
):
    """
    Evaluate adversarial examples for the supplied attack-eligible
    observations only.
    """

    y_true = np.asarray(y_true)
    clean_predictions = np.asarray(clean_predictions)
    clean_probabilities = np.asarray(clean_probabilities)

    attacked_predictions = np.asarray(attacked_predictions)
    attacked_probabilities = np.asarray(attacked_probabilities)

    X_original = np.asarray(X_original, dtype=float)
    X_attacked = np.asarray(X_attacked, dtype=float)

    n = len(y_true)

    if len(clean_predictions) != n:
        raise ValueError("clean_predictions length mismatch.")

    if len(clean_probabilities) != n:
        raise ValueError("clean_probabilities length mismatch.")

    if len(attacked_predictions) != n:
        raise ValueError("attacked_predictions length mismatch.")

    if len(attacked_probabilities) != n:
        raise ValueError("attacked_probabilities length mismatch.")

    if X_original.shape != X_attacked.shape:
        raise ValueError(
            "X_original and X_attacked must have identical shapes."
        )

    eligible = (
        (y_true == 1)
        & (clean_predictions == 1)
    )

    n_eligible = int(np.sum(eligible))

    if n_eligible == 0:
        raise ValueError(
            "No clean-correct malicious samples are available."
        )

    successful = (
        eligible
        & (attacked_predictions == 0)
    )

    n_successful = int(np.sum(successful))

    asr = n_successful / n_eligible

    perturbation = (
        X_attacked[eligible]
        - X_original[eligible]
    )

    absolute_perturbation = np.abs(perturbation)

    changed_features = np.count_nonzero(
        absolute_perturbation > 1e-12,
        axis=1,
    )

    l2_distance = np.linalg.norm(
        perturbation,
        axis=1,
    )

    linf_distance = np.max(
        absolute_perturbation,
        axis=1,
    )

    confidence_drop = (
        clean_probabilities[eligible]
        - attacked_probabilities[eligible]
    )

    return {
        "eligible_samples": n_eligible,
        "successful_evasions": n_successful,
        "attack_success_rate": float(asr),
        "mean_changed_features": float(
            np.mean(changed_features)
        ),
        "mean_l2": float(
            np.mean(l2_distance)
        ),
        "mean_linf": float(
            np.mean(linf_distance)
        ),
        "mean_clean_malicious_probability": float(
            np.mean(clean_probabilities[eligible])
        ),
        "mean_attacked_malicious_probability": float(
            np.mean(attacked_probabilities[eligible])
        ),
        "mean_confidence_drop": float(
            np.mean(confidence_drop)
        ),
    }
def run_fgsm_mlp(
    epsilon: float,
    sample_limit: int | None = None,
):
    """
    Run a controlled FGSM experiment against the frozen MLP.

    Only clean-correct malicious observations are attacked.
    """

    X_test, y_test = load_unsw_test_data()

    model = load_model("mlp")

    clean_predictions, clean_probabilities = (
        get_clean_predictions(
            model,
            X_test,
        )
    )

    eligible_indices = select_attack_eligible_samples(
        y_test,
        clean_predictions,
    )

    if sample_limit is not None:
        eligible_indices = eligible_indices[:sample_limit]

    X_attack = X_test[eligible_indices]
    y_attack = y_test[eligible_indices]

    clean_attack_predictions = clean_predictions[
        eligible_indices
    ]

    clean_attack_probabilities = clean_probabilities[
        eligible_indices
    ]

    X_adv = conventional_attacks.fgsm_attack_mlp(
        model,
        X_attack,
        y_attack,
        epsilon=epsilon,
    )

    attacked_predictions, attacked_probabilities = (
        get_clean_predictions(
            model,
            X_adv,
        )
    )

    results = evaluate_adversarial_examples(
        y_true=y_attack,
        clean_predictions=clean_attack_predictions,
        clean_probabilities=clean_attack_probabilities,
        attacked_predictions=attacked_predictions,
        attacked_probabilities=attacked_probabilities,
        X_original=X_attack,
        X_attacked=X_adv,
    )

    results["model"] = "mlp"
    results["attack"] = "fgsm"
    results["epsilon"] = epsilon

    return results


if __name__ == "__main__":
    results = run_fgsm_mlp(
        epsilon=0.01,
        sample_limit=100,
    )

    print("\nFGSM test run")
    print("=" * 50)

    for key, value in results.items():
        print(f"{key}: {value}")