
from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "attacks"
    / "06_conventional_attacks.py"
)

spec = importlib.util.spec_from_file_location(
    "conventional_attacks",
    MODULE_PATH,
)

conventional_attacks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conventional_attacks)


def main():
    model = joblib.load(
        PROJECT_ROOT / "results" / "models" / "mlp_clean.joblib"
    )

    X_test = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "unsw_X_test.csv"
    )

    y_test = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "unsw_y_test.csv"
    ).squeeze("columns")

    # Use only a small sample for the sanity check.
    X_sample = X_test.iloc[:100].to_numpy(dtype=float)
    y_sample = y_test.iloc[:100].to_numpy()

    results = conventional_attacks.test_mlp_gradient(
        model,
        X_sample,
        y_sample,
    )

    print("\nMLP gradient sanity check")
    print("-" * 40)

    for key, value in results.items():
        print(f"{key}: {value}")


def main():
    model = joblib.load(
        PROJECT_ROOT / "results" / "models" / "mlp_clean.joblib"
    )

    X_test = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "unsw_X_test.csv"
    )

    y_test = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "unsw_y_test.csv"
    ).squeeze("columns")

    X_sample = X_test.iloc[:100].to_numpy(dtype=float)
    y_sample = y_test.iloc[:100].to_numpy()

    # Gradient sanity check
    gradient_results = conventional_attacks.test_mlp_gradient(
        model,
        X_sample,
        y_sample,
    )

    print("\nMLP gradient sanity check")
    print("-" * 40)

    for key, value in gradient_results.items():
        print(f"{key}: {value}")

    # Small FGSM sanity check
    epsilon = 0.01

    X_adv = conventional_attacks.fgsm_attack_mlp(
        model,
        X_sample,
        y_sample,
        epsilon=epsilon,
    )

    perturbation = X_adv - X_sample

    print("\nFGSM sanity check")
    print("-" * 40)
    print(f"epsilon: {epsilon}")
    print(f"shape: {X_adv.shape}")
    print(f"maximum absolute change: {np.max(np.abs(perturbation))}")
    print(
        f"mean absolute change: {np.mean(np.abs(perturbation))}"
    )
    print(
        f"changed feature fraction: "
        f"{np.mean(np.abs(perturbation) > 1e-12)}"
    )


if __name__ == "__main__":
    main()