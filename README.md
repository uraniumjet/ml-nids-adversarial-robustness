# Evaluating the Adversarial Robustness and Trustworthiness of Machine-Learning-Based Network Intrusion Detection under Realistic Attack Constraints

## Research Project

This repository contains the experimental implementation for a research study investigating the adversarial robustness and predictive trustworthiness of machine-learning-based network intrusion detection systems (ML-NIDS) under progressively realistic attack constraints.

## Research Questions

RQ1. How does the measured adversarial robustness of machine-learning-based NIDS change when adversarial attacks are evaluated under realistic constraints rather than unrestricted feature-space perturbations?

RQ2. How do different machine-learning models compare in detection performance, adversarial attack success, robustness degradation, and false-positive behaviour under conventional and constraint-aware adversarial attacks?

RQ3. To what extent does adversarial evasion affect the predictive confidence and probability calibration of machine-learning-based NIDS?

RQ4. Does the choice of adversarial evaluation methodology change the relative ranking of machine-learning models in terms of adversarial robustness?

RQ5. Which evaluation factors and experimental practices are most important for producing reproducible and practically meaningful assessments of adversarial robustness in ML-based NIDS?

## Datasets

The primary datasets are:

- UNSW-NB15
- CIC-IDS2017

Raw datasets are not stored in this repository.

## Models

The planned baseline models are:

- Logistic Regression
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

## Evaluation Conditions

1. Clean baseline
2. Conventional feature-space adversarial attacks
3. Constraint-aware adversarial attacks

## Trustworthiness Measures

- Expected Calibration Error (ECE)
- Brier score
- Negative Log-Likelihood (NLL)
- Predictive confidence
- High-confidence successful evasions

## Reproducibility

All preprocessing procedures, model configurations, attack parameters, random seeds, evaluation metrics, and statistical procedures will be documented as the experiments are developed.

## Status

Research methodology completed.

Experimental implementation in progress.
