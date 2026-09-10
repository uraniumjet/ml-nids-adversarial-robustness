from __future__ import annotations

import numpy as np


def get_numerical_feature_mask(n_features: int = 192) -> np.ndarray:
    """
    Return a boolean mask for the 39 numerical UNSW-NB15
    processed features.

    Current preprocessing produces:
        39 numerical features
        153 one-hot categorical features
        192 total features
    """
    if n_features != 192:
        raise ValueError(
            f"Expected 192 processed features, got {n_features}."
        )

    mask = np.zeros(n_features, dtype=bool)
    mask[:39] = True
    return mask


def clip_to_feature_bounds(
    X: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Clip each feature to its supplied lower/upper bound."""
    return np.minimum(np.maximum(X, lower_bounds), upper_bounds)

def mlp_input_gradient(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Calculate d(loss)/d(X) for a fitted sklearn MLPClassifier.

    Supports an arbitrary number of ReLU hidden layers and a
    binary sigmoid output.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if len(y) != len(X):
        raise ValueError("X and y must contain the same number of samples.")

    # Forward pass.
    activations = [X]
    pre_activations = []

    A = X

    # Hidden layers.
    for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        Z = A @ W + b
        pre_activations.append(Z)

        # sklearn MLP uses ReLU for this model.
        A = np.maximum(Z, 0.0)
        activations.append(A)

    # Output layer.
    W_out = model.coefs_[-1]
    b_out = model.intercepts_[-1]

    logits = activations[-1] @ W_out + b_out

    if logits.shape[1] != 1:
        raise ValueError(
            f"Expected binary output with shape (n, 1), got {logits.shape}."
        )

    logits = logits[:, 0]

    # Stable sigmoid.
    logits = np.clip(logits, -50.0, 50.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    # Binary cross-entropy:
    #
    # dL/d(logit) = sigmoid(logit) - y
    d_logits = probabilities - y

    # Backpropagate through output layer.
    dA = d_logits[:, None] @ W_out.T

    # Backpropagate through every hidden layer.
    for layer in reversed(range(len(pre_activations))):

        Z = pre_activations[layer]

        # Derivative of ReLU.
        dZ = dA * (Z > 0.0)

        # Gradient with respect to the previous activation.
        W = model.coefs_[layer]
        dA = dZ @ W.T

    # dA is now d(loss)/d(input).
    gradient = dA

    if gradient.shape != X.shape:
        raise RuntimeError(
            f"Gradient shape {gradient.shape} does not match X shape {X.shape}."
        )

    return gradient
def fgsm_attack_mlp(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    feature_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    FGSM attack for the MLP.

    The attack maximizes binary cross-entropy loss.
    """
    X = np.asarray(X, dtype=float)

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    gradient = mlp_input_gradient(model, X, y)

    perturbation = epsilon * np.sign(gradient)

    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask, dtype=bool)

        if feature_mask.shape[0] != X.shape[1]:
            raise ValueError("feature_mask length does not match X.")

        perturbation[:, ~feature_mask] = 0.0

    return X + perturbation


def pgd_attack_mlp(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    step_size: float,
    num_steps: int,
    feature_mask: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> np.ndarray:
    """
    Projected Gradient Descent attack.

    The adversarial example remains inside the L-infinity epsilon ball
    around the original input.

    Optional feature bounds can additionally restrict each feature.
    """
    X = np.asarray(X, dtype=float)

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    if feature_mask is None:
        feature_mask = np.ones(X.shape[1], dtype=bool)
    else:
        feature_mask = np.asarray(feature_mask, dtype=bool)

    if feature_mask.shape[0] != X.shape[1]:
        raise ValueError("feature_mask length does not match X.")

    X_adv = X.copy()

    for _ in range(num_steps):
        gradient = mlp_input_gradient(model, X_adv, y)

        perturbation = step_size * np.sign(gradient)

        # Only perturb permitted features.
        perturbation[:, ~feature_mask] = 0.0

        X_adv = X_adv + perturbation

        # Project into L-infinity epsilon ball.
        delta = np.clip(
            X_adv - X,
            -epsilon,
            epsilon,
        )

        delta[:, ~feature_mask] = 0.0

        X_adv = X + delta

        # Optional feature-wise bounds.
        if lower_bounds is not None and upper_bounds is not None:
            X_adv = clip_to_feature_bounds(
                X_adv,
                lower_bounds,
                upper_bounds,
            )

            # Ensure non-attacked features remain unchanged.
            X_adv[:, ~feature_mask] = X[:, ~feature_mask]

    return X_adv


def get_attack_eligible_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Return indices of malicious samples that were correctly detected
    by the clean model.

    These are the only samples eligible for an evasion attack.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.where(
        (y_true == 1) &
        (y_pred == 1)
    )[0]


def get_positive_class_probability(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """Return probability assigned to class 1."""
    probabilities = model.predict_proba(X)
    class_index = np.where(model.classes_ == 1)[0]

    if len(class_index) != 1:
        raise ValueError("Could not identify class 1.")

    return probabilities[:, class_index[0]]


def evaluate_adversarial_examples(
    model,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Evaluate adversarial examples.

    Successful evasion:
        y_true == 1
        clean prediction == 1
        adversarial prediction == 0
    """
    y_true = np.asarray(y_true)

    clean_pred = model.predict(X_clean)
    adv_pred = model.predict(X_adv)

    eligible = (
        (y_true == 1) &
        (clean_pred == 1)
    )

    if not np.any(eligible):
        return {
            "eligible_samples": 0,
            "successful_evasions": 0,
            "attack_success_rate": np.nan,
            "mean_changed_features": np.nan,
            "mean_l2": np.nan,
            "mean_linf": np.nan,
            "mean_clean_malicious_probability": np.nan,
            "mean_attacked_malicious_probability": np.nan,
            "mean_confidence_drop": np.nan,
        }

    clean_eval = X_clean[eligible]
    adv_eval = X_adv[eligible]

    successful = adv_pred[eligible] == 0

    delta = adv_eval - clean_eval

    changed = np.sum(
        ~np.isclose(delta, 0.0),
        axis=1,
    )

    l2 = np.linalg.norm(delta, axis=1)
    linf = np.max(np.abs(delta), axis=1)

    clean_probability = get_positive_class_probability(
        model,
        clean_eval,
    )

    attacked_probability = get_positive_class_probability(
        model,
        adv_eval,
    )

    confidence_drop = (
        clean_probability - attacked_probability
    )

    return {
        "eligible_samples": int(np.sum(eligible)),
        "successful_evasions": int(np.sum(successful)),
        "attack_success_rate": float(np.mean(successful)),
        "mean_changed_features": float(np.mean(changed)),
        "mean_l2": float(np.mean(l2)),
        "mean_linf": float(np.mean(linf)),
        "mean_clean_malicious_probability": float(
            np.mean(clean_probability)
        ),
        "mean_attacked_malicious_probability": float(
            np.mean(attacked_probability)
        ),
        "mean_confidence_drop": float(
            np.mean(confidence_drop)
        ),
    }

def test_mlp_gradient(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Run a basic sanity check on the MLP input gradient.
    """
    gradient = mlp_input_gradient(model, X, y)

    return {
        "shape": gradient.shape,
        "min": float(np.min(gradient)),
        "max": float(np.max(gradient)),
        "mean_absolute": float(np.mean(np.abs(gradient))),
        "nonzero_fraction": float(
            np.mean(gradient != 0)
        ),
    }

def test_mlp_gradient(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """
    Run a basic sanity check on the MLP input gradient.
    """
    gradient = mlp_input_gradient(model, X, y)

    return {
        "shape": gradient.shape,
        "min": float(np.min(gradient)),
        "max": float(np.max(gradient)),
        "mean_absolute": float(np.mean(np.abs(gradient))),
        "nonzero_fraction": float(
            np.mean(gradient != 0)
        ),
    }