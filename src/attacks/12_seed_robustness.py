"""
Multi-seed robustness of the main result (answers reviewer comments on single run / single seed).

Retrains the MLP with the paper's hyper-parameters for several random seeds. For every seed it
draws its own 1,000-observation cohort (as the paper does for seed 42), then runs configurations
  A  all 39 numerical features, no clipping         (conventional)
  B  all 39, clipping
  C  27 mutable, no clipping
  D  27 mutable, clipping                           (constraint-aware)
with FGSM and PGD at eps 0.05/0.10/0.20, and reports mean, SD and 95% CI across seeds.

BUILT-IN CHECK: seed 42 must reproduce the paper's clean MLP test accuracy 0.8512 and FPR 0.3039.
If it does not, the training data or hyper-parameters differ from the paper; stop and tell me.

Edit N_SEEDS / SEEDS below if training takes too long. Run:
    python src/attacks/12_seed_robustness.py
"""
from pathlib import Path
import importlib.util, json, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.neural_network import MLPClassifier

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)

SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y_TRAIN_PATH = abl.ROOT / "data/processed/unsw_y_train.csv"      # edit if your file is named differently
PAPER_ACC, PAPER_FPR = 0.8512, 0.3039


def train_mlp(X, y, seed):
    return MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                         learning_rate_init=0.001, batch_size=256, alpha=0.0001, max_iter=50,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
                         random_state=seed).fit(X, y)


def clean_metrics(ca, model, X_test, y_test):
    p = ca.get_positive_class_probability(model, X_test); pred = (p >= 0.5).astype(int)
    return {"accuracy": float((pred == y_test).mean()),
            "fpr": float(((pred == 1) & (y_test == 0)).sum() / (y_test == 0).sum()),
            "fnr": float(((pred == 0) & (y_test == 1)).sum() / (y_test == 1).sum())}, p


def evaluate_seed(ca, model, p_test, X_test, y_test, lower, upper, names, seed, n_samples=abl.N_SAMPLES):
    numerical, mut27, _ = abl.build_masks(names)
    eligible = np.where((y_test == 1) & (p_test >= 0.5))[0]
    idx = np.sort(np.random.default_rng(seed).choice(eligible, size=min(n_samples, len(eligible)), replace=False))
    X, y = X_test[idx].copy(), y_test[idx].copy()
    rows = []
    for cfg, mask, clip in (("A", numerical, False), ("B", numerical, True), ("C", mut27, False), ("D", mut27, True)):
        for kind in ("FGSM", "PGD"):
            for eps in abl.EPSILONS:
                Xa = abl.attack_once(ca, model, X, y, eps, kind, mask, lower, upper, clip)
                m = abl.metrics(ca, model, X, Xa, y)
                rows.append({"seed": seed, "config": cfg, "attack": kind, "epsilon": eps,
                             "asr": m["asr"], "mean_conf_drop": m["mean_conf_drop"], "n_cohort": len(idx)})
    return rows


def summarise(res):
    def ci(v):
        v = np.asarray(v, float); n = len(v)
        if n < 2: return (np.nan, np.nan)
        h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n); return (v.mean() - h, v.mean() + h)
    out = []
    for (cfg, kind, eps), g in res.groupby(["config", "attack", "epsilon"]):
        lo, hi = ci(g.asr); out.append({"config": cfg, "attack": kind, "epsilon": eps, "n_seeds": len(g),
                                        "mean": g.asr.mean(), "sd": g.asr.std(ddof=1), "min": g.asr.min(),
                                        "max": g.asr.max(), "ci_lo": lo, "ci_hi": hi})
    s = pd.DataFrame(out)
    # paired difference A - D per seed
    piv = res.pivot_table(index=["seed", "attack", "epsilon"], columns="config", values="asr").reset_index()
    piv["diff_A_minus_D"] = piv["A"] - piv["D"]
    d = []
    for (kind, eps), g in piv.groupby(["attack", "epsilon"]):
        lo, hi = ci(g.diff_A_minus_D); d.append({"attack": kind, "epsilon": eps, "n_seeds": len(g),
                                                  "mean_diff": g.diff_A_minus_D.mean(), "sd": g.diff_A_minus_D.std(ddof=1),
                                                  "ci_lo": lo, "ci_hi": hi, "min_diff": g.diff_A_minus_D.min()})
    return s, pd.DataFrame(d)


def main():
    warnings.filterwarnings("ignore")
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values; y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    X_test = pd.read_csv(abl.X_TEST_PATH).values; y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    import joblib
    names = abl.build_feature_names(meta, joblib.load(abl.PREP_PATH), X_test.shape[1])
    lower, upper = X_train.min(axis=0), X_train.max(axis=0)
    print(f"Training data: {X_train.shape}, labels: {y_train.shape}")
    rows, clean = [], []
    for seed in SEEDS:
        t = time.time(); model = train_mlp(X_train, y_train, seed)
        cm, p = clean_metrics(ca, model, X_test, y_test); cm["seed"] = seed; clean.append(cm)
        print(f"seed {seed}: acc={cm['accuracy']:.4f} fpr={cm['fpr']:.4f} (train {time.time()-t:.0f}s)", flush=True)
        if seed == 42:
            ok = abs(cm["accuracy"] - PAPER_ACC) < 5e-4 and abs(cm["fpr"] - PAPER_FPR) < 5e-4
            print("  REPRODUCTION CHECK vs paper (acc 0.8512, FPR 0.3039):", "OK" if ok else "MISMATCH - stop and check training data / settings")
        rows += evaluate_seed(ca, model, p, X_test, y_test, lower, upper, names, seed)
    res = pd.DataFrame(rows); res.to_csv(abl.OUTPUT_DIR / "seed_robustness.csv", index=False)
    cl = pd.DataFrame(clean); cl.to_csv(abl.OUTPUT_DIR / "seed_clean_metrics.csv", index=False)
    s, d = summarise(res)
    s.to_csv(abl.OUTPUT_DIR / "seed_summary.csv", index=False); d.to_csv(abl.OUTPUT_DIR / "seed_diff_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nCLEAN MLP across seeds: accuracy mean %.4f SD %.4f | FPR mean %.4f SD %.4f" %
          (cl.accuracy.mean(), cl.accuracy.std(), cl.fpr.mean(), cl.fpr.std()))
    print("\nASR across seeds at eps=0.20 (mean, SD, 95% CI)")
    print(s[s.epsilon == 0.20].round(3).to_string(index=False))
    print("\nConventional (A) minus constraint-aware (D), paired by seed")
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()