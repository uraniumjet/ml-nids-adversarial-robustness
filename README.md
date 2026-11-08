# Attack-space assumptions and measured robustness of ML-based network intrusion detection (UNSW-NB15)

Code, configuration and result files for the paper

> **Evaluating Adversarial Robustness and Predictive Trustworthiness of Machine-Learning-Based Network Intrusion Detection Under Constrained Feature-Space Attacks**
> Yusuf Ahmad Shehu, Ayaz Ahmed Faridi — Kalinga University, Raipur, India.
> An earlier version was presented at ICDIACS 2026.

## What the study does

A multilayer perceptron (MLP) trained on UNSW-NB15 is attacked with FGSM and PGD (white-box, feature space) in two attack spaces, with the model, the cohort of **1,000 correctly detected malicious test observations** (seed 42), the budgets (ε = 0.05, 0.10, 0.20 in standardized units) and the preprocessing held fixed:

* **conventional**: all 39 numerical features mutable, no clipping;
* **constraint-aware**: 27 mutable numerical features, clipped to the training min/max; 12 contextual features and 153 one-hot dimensions fixed.

Further analyses: component ablation and random-27 control, mask sensitivity, integer rounding, Brier score and ECE under attack, a query-based attack on Random Forest and XGBoost, 11 retrained MLPs, paired tests, attack-category breakdown, and a train–test overlap audit.

**Headline result (ε = 0.20, saved MLP):** attack success rate FGSM 34.7% → 8.8% and PGD 41.5% → 13.6% when the attack space is constrained; over 11 retrained MLPs the means are 34.0% → 15.0% and 39.6% → 16.6%. The study is a controlled case study of one dataset, one architecture and feature-space attacks. It does **not** implement a host-space attack, and the constraint-aware setting must not be read as a realistic attacker.

## Repository layout

Script numbers are only unique within a folder.

## Which script produces what

| Output in the paper | Script (folder `src/`) | Result files |
|---|---|---|
| Clean baselines (Table 2) | `models/03_train_clean_baseline.py` | `results/models/` (only the MLP is tracked, see below) |
| MLP FGSM/PGD, conventional (Table 3) | `attacks/10_main_conventional_experiment.py` | `conventional_attack_results.csv` |
| MLP FGSM/PGD, constraint-aware (Table 4) | `attacks/11_constraint_aware_attacks.py` | `constraint_aware_attack_results_mlp.csv` |
| Bootstrap effect sizes (Table 6) | `analysis/14_effect_size_bootstrap.py` | `results/analysis/effect_size_bootstrap_results_mlp.csv` |
| Full-test Brier/ECE (Table 5) | `analysis/15_full_test_trustworthiness.py` | `results/analysis/` |
| Ablation, random-27, unfreezing (Tables 7, 8) | `attacks/08_constraint_ablation.py` | `ablation_*.csv`, `bound_share.csv`, `gradient_importance.csv` |
| Mask sensitivity, rounding (Table 9) | `attacks/09_mask_sensitivity.py` | `mask_sensitivity.csv` |
| Balanced / attacked-set Brier and ECE (Table 10) | `attacks/10_trust_metrics.py` | `trust_metrics.csv` |
| Tree models, query-based attack (Table 11) | `attacks/11_tree_blackbox.py` | `tree_blackbox_results.csv` |
| 11 seeds (Tables 12, 13) | `attacks/12_seed_robustness.py` | `seed_*.csv` |
| Paired tests, perturbation size (Tables 14, 15) | `attacks/13_stats_and_perturbation.py` | `paired_tests.csv`, `perturbation_summary.csv` |
| Cohort composition, categories (Tables 16, 17) | `attacks/14_cohort_composition.py` | `cohort_composition.csv`, `asr_by_category.csv` |
| Train–test overlap (Tables 18, 19) | `audit/15_env_and_duplicates.py`, `audit/16_overlap_sensitivity.py` | `overlap_*.csv` |
| Attack cohort (row indices into the test set) | `attacks/17_export_cohort.py` | `cohort_indices.csv` |

`attacks/10_main_conventional_experiment.py` also contains an early black-box attack for Random Forest and XGBoost that searched only the first 10 numerical features. **That part is superseded by `11_tree_blackbox.py`** and its output is not used in the paper.

## Reproducing

1. Obtain UNSW-NB15 (training and testing partition CSV files) from <https://research.unsw.edu.au/projects/unsw-nb15-dataset> and place them in `data/raw/` (the data are not redistributed).
2. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Run the scripts in the order of the folders above (`data` → `models` → `attacks` → `analysis` → `audit`).

Environment used: Python 3.14.4, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.0, XGBoost 3.4.1; CPU only. `requirements-freeze.txt` is the full `pip freeze` of that environment.

**Reproducibility note.** The main-analysis numbers come from one saved trained MLP (`results/models/mlp_clean.joblib`, scikit-learn 1.9.0, `random_state=42`). Retraining with the same settings gives slightly different weights and therefore slightly different numbers (clean accuracy 0.8512 for the saved model; 0.858 ± 0.009 over 11 seeds). The saved model is included so the main tables can be reproduced exactly; loading a joblib file executes code, so only load it from a source you trust. The seed analysis (`12_seed_robustness.py`) shows how much the results vary between trainings.

## Known limitations

One dataset; one white-box target (MLP); feature-space attacks only; PGD without random starts or restarts; tree-model attacks use a weaker query-based search and are not a matched comparison; UNSW-NB15 has train/test overlap (10.4% of test rows have a training twin), analysed in `audit/`.

## Citation and licence

See `CITATION.cff`. Code: MIT licence (`LICENSE`). The UNSW-NB15 dataset has its own terms; see the link above.
