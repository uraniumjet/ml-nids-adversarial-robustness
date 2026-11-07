"""
Corrected black-box coordinate attack for Random Forest / XGBoost (answers reviewer comment R51).

PROBLEM FOUND in 10_main_conventional_experiment.py::black_box_coordinate_attack:
    feature_indices = feature_indices[:features_per_step]
takes the FIRST 10 numerical features, the same ones for every sample, on every iteration.
So the original tree results only ever perturbed 10 of the 39 features (not "all numerical
dimensions" as the comment says).  This script:
  orig_first10    reproduces the original behaviour (must give 0.0131/0.0172/0.0394 for RF,
                  0.1463/0.1593/0.1633 for XGBoost) -> confirms the diagnosis
  conv_all39      same coordinate search over ALL 39 numerical features, no clipping
  constrained_27  search over the 27 mutable features + clipping to training bounds
Same cohort, same epsilons, same seed as the MLP experiments.  Place next to 08_*.py and run:
    python src/attacks/11_tree_blackbox.py
If the model files are not found automatically, edit MODEL_FILES below.
"""
from pathlib import Path
import importlib.util, json
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)

STEPS = 10
MODEL_FILES = {}          # e.g. {"randomforest": abl.ROOT/"results/models/randomforest_clean.joblib", "xgboost": ...}


def discover_models():
    if MODEL_FILES: return MODEL_FILES
    found = {}
    for f in sorted((abl.ROOT / "results/models").glob("*.joblib")):
        n = f.stem.lower()
        if "random" in n or n.startswith("rf"): found.setdefault("randomforest", f)
        if "xgb" in n or "xgboost" in n: found.setdefault("xgboost", f)
    print("Models found:", {k: str(v.name) for k, v in found.items()})
    if len(found) < 2: raise SystemExit("Could not find both models; set MODEL_FILES at the top of this script.")
    return found


def coordinate_attack(ca, model, X, eps, feature_idx, steps=STEPS, lower=None, upper=None, clip=False):
    """Greedy +/- coordinate search; identical logic to the original, with the feature list explicit."""
    Xa = X.copy(); step = eps / steps
    p_cur = ca.get_positive_class_probability(model, Xa)
    for _ in range(steps):
        for j in feature_idx:
            cur = Xa[:, j].copy()
            lo_j, hi_j = X[:, j] - eps, X[:, j] + eps
            if clip: lo_j = np.maximum(lo_j, lower[j]); hi_j = np.minimum(hi_j, upper[j])
            plus, minus = Xa.copy(), Xa.copy()
            plus[:, j] = np.clip(cur + step, lo_j, hi_j); minus[:, j] = np.clip(cur - step, lo_j, hi_j)
            pp = ca.get_positive_class_probability(model, plus)
            pm = ca.get_positive_class_probability(model, minus)
            use_plus = pp < p_cur
            use_minus = (pm < p_cur) & (pm <= pp)
            Xa[use_plus, j] = plus[use_plus, j]; Xa[use_minus, j] = minus[use_minus, j]
            p_cur = np.where(use_plus, pp, np.where(use_minus, pm, p_cur))
    return Xa


def run(ca, models, X_attack, y_attack, lower, upper, names, epsilons=abl.EPSILONS, steps=STEPS):
    numerical, mut27, _ = abl.build_masks(names)
    num_idx, mut_idx = np.flatnonzero(numerical), np.flatnonzero(mut27)
    configs = {"orig_first10": (num_idx[:10], False), "conv_all39": (num_idx, False), "constrained_27": (mut_idx, True)}
    rows = []
    for mname, model in models.items():
        p0 = ca.get_positive_class_probability(model, X_attack)
        el = (y_attack == 1) & (p0 >= 0.5)
        X, p_clean = X_attack[el], p0[el]
        for cname, (fidx, clip) in configs.items():
            for eps in epsilons:
                Xa = coordinate_attack(ca, model, X, eps, fidx, steps, lower, upper, clip)
                p1 = ca.get_positive_class_probability(model, Xa)
                rows.append({"model": mname, "config": cname, "epsilon": eps, "eligible_n": int(el.sum()),
                             "successes": int((p1 < 0.5).sum()), "asr": float((p1 < 0.5).mean()),
                             "mean_conf_drop": float((p_clean - p1).mean()),
                             "mean_changed_features": float((np.abs(Xa - X) > 1e-12).sum(1).mean()),
                             "mean_l2": float(np.linalg.norm(Xa - X, axis=1).mean()),
                             "features_searched": len(fidx), "queries_per_sample": steps * len(fidx) * 2})
                print(mname, cname, eps, rows[-1]["asr"], flush=True)
    return pd.DataFrame(rows)


def main():
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values
    X_test = pd.read_csv(abl.X_TEST_PATH).values
    y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    mlp = joblib.load(abl.MODEL_PATH)
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    names = abl.build_feature_names(meta, joblib.load(abl.PREP_PATH), X_test.shape[1])
    lower, upper = X_train.min(axis=0), X_train.max(axis=0)
    p = ca.get_positive_class_probability(mlp, X_test)
    eligible = np.where((y_test == 1) & (p >= 0.5))[0]
    idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
    X_attack, y_attack = X_test[idx].copy(), y_test[idx].copy()
    numerical, _, _ = abl.build_masks(names)
    print("First 10 numerical features (the ones the ORIGINAL attack used):",
          [names[i].replace("numerical__", "") for i in np.flatnonzero(numerical)[:10]])
    models = {k: joblib.load(v) for k, v in discover_models().items()}
    res = run(ca, models, X_attack, y_attack, lower, upper, names)
    res.to_csv(abl.OUTPUT_DIR / "tree_blackbox_results.csv", index=False)
    print("\nREPRODUCTION of original tree results (config orig_first10):")
    print(res[res.config == "orig_first10"][["model", "epsilon", "eligible_n", "successes", "asr"]].round(4).to_string(index=False))
    print("\nALL RESULTS:")
    print(res[["model", "config", "epsilon", "eligible_n", "successes", "asr", "mean_conf_drop",
               "mean_changed_features", "mean_l2", "features_searched"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()