from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

def get_attack_eligible_indices(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Return indices of malicious samples that were correctly detected
    by the clean model.

    These samples are eligible for adversarial evasion evaluation.

    Assumptions:
    - label 0 = benign
    - label 1 = malicious
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    eligible = (y_true == 1) & (y_pred == 1)

    return np.flatnonzero(eligible)
def calculate_attack_success_rate(
    y_true: pd.Series | np.ndarray,
    clean_pred: pd.Series | np.ndarray,
    attacked_pred: pd.Series | np.ndarray,
) -> float:
    """
    Calculate adversarial attack success rate.

    ASR is calculated only among malicious samples that were
    correctly classified as malicious by the clean model.

    A successful attack changes the prediction from:
        malicious (1) -> benign (0)
    """

    y_true = np.asarray(y_true)
    clean_pred = np.asarray(clean_pred)
    attacked_pred = np.asarray(attacked_pred)

    if not (
        len(y_true) == len(clean_pred) == len(attacked_pred)
    ):
        raise ValueError(
            "y_true, clean_pred, and attacked_pred must have the same length."
        )

    eligible = (y_true == 1) & (clean_pred == 1)

    n_eligible = np.sum(eligible)

    if n_eligible == 0:
        return np.nan

    successful_evasions = np.sum(
        eligible & (attacked_pred == 0)
    )

    return successful_evasions / n_eligible

def calculate_perturbation_metrics(
    X_original: np.ndarray,
    X_attacked: np.ndarray,
) -> dict:
    """
    Calculate basic perturbation statistics between original
    and adversarial feature representations.

    Returns:
        l0_mean: mean number of changed features
        l2_mean: mean Euclidean perturbation
        linf_mean: mean maximum absolute feature change
    """

    X_original = np.asarray(X_original, dtype=float)
    X_attacked = np.asarray(X_attacked, dtype=float)

    if X_original.shape != X_attacked.shape:
        raise ValueError(
            "X_original and X_attacked must have the same shape."
        )

    difference = X_attacked - X_original
    absolute_difference = np.abs(difference)

    changed_features = np.count_nonzero(
        absolute_difference > 1e-12,
        axis=1,
    )

    l2_distance = np.linalg.norm(
        difference,
        ord=2,
        axis=1,
    )

    linf_distance = np.max(
        absolute_difference,
        axis=1,
    )

    return {
        "l0_mean": float(np.mean(changed_features)),
        "l2_mean": float(np.mean(l2_distance)),
        "linf_mean": float(np.mean(linf_distance)),
                }

def get_positive_class_probability(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return the model probability for the malicious class (label 1).

    The model must expose predict_proba().
    """

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The supplied model must implement predict_proba()."
        )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected predict_proba() to return probabilities for "
            "at least two classes."
        )

    return probabilities[:, 1]