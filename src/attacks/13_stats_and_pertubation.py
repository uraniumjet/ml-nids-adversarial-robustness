"""
Statistics table + perturbation table for the main MLP (answers reviewer comments R40 and R59).

For FGSM/PGD at eps 0.05/0.10/0.20, conventional (39 features, no clipping) versus
constraint-aware (27 mutable, clipping), paired per observation on the 1,000 cohort:
  McNemar exact test on the evasion outcomes (discordant pairs b and c, two-sided p)
  Wilcoxon signed-rank test on the confidence drops, rank-biserial effect size
  Holm correction across the 6 comparisons, separately for each test
And the perturbation size of each attack: mean changed features, mean L2, mean Linf.

Uses the SAVED mlp_clean.joblib, the same cohort as the paper. Place next to 08_*.py; run:
    python src/attacks/13_stats_and_perturbation.py
"""
from pathlib import Path
import importlib.util, json
import numpy as np, pandas as pd, joblib
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0: return np.nan
    r = stats.rankdata(np.abs(d)); pos = r[d > 0].sum(); neg = r[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def pert(X, Xa):
    D = Xa - X
    return {"mean_changed_features": float((np.abs(D) > 1e-12).sum(1).mean()),
            "mean_l2": float(np.linalg.norm(D, axis=1).mean()), "mean_linf": float(np.abs(D).max(1).mean())}


def compute(ca, model, X, y, lower, upper, names, epsilons=abl.EPSILONS):
    numerical, mut27, _ = abl.build_masks(names)
    p0 = ca.get_positive_class_probability(model, X)
    rows, prow = [], []
    for kind in ("FGSM", "PGD"):
        for eps in epsilons:
            Xc = abl.attack_once(ca, model, X, y, eps, kind, numerical, lower, upper, False)
            Xk = abl.attack_once(ca, model, X, y, eps, kind, mut27, lower, upper, True)
            pc, pk = ca.get_positive_class_probability(model, Xc), ca.get_positive_class_probability(model, Xk)
            sc, sk = pc < 0.5, pk < 0.5
            b = int((sc & ~sk).sum()); c = int((~sc & sk).sum())
            mc = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
            dc, dk = p0 - pc, p0 - pk; diff = dc - dk
            w = stats.wilcoxon(diff, zero_method="wilcox") if np.any(diff != 0) else None
            rows.append({"attack": kind, "epsilon": eps, "asr_conv": sc.mean(), "asr_constr": sk.mean(),
                         "b_conv_only": b, "c_constr_only": c, "mcnemar_p": mc,
                         "drop_conv": dc.mean(), "drop_constr": dk.mean(), "drop_diff": diff.mean(),
                         "wilcoxon_stat": w.statistic if w else np.nan, "wilcoxon_p": w.pvalue if w else 1.0,
                         "rank_biserial": rank_biserial(diff)})
            for cond, Xa in (("conventional", Xc), ("constraint-aware", Xk)):
                prow.append({"attack": kind, "epsilon": eps, "condition": cond, **pert(X, Xa)})
    r = pd.DataFrame(rows)
    r["mcnemar_p_holm"] = holm(r.mcnemar_p); r["wilcoxon_p_holm"] = holm(r.wilcoxon_p)
    return r, pd.DataFrame(prow)


def main():
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values
    X_test = pd.read_csv(abl.X_TEST_PATH).values; y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    model = joblib.load(abl.MODEL_PATH)
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    names = abl.build_feature_names(meta, joblib.load(abl.PREP_PATH), X_test.shape[1])
    lower, upper = X_train.min(axis=0), X_train.max(axis=0)
    p = ca.get_positive_class_probability(model, X_test)
    eligible = np.where((y_test == 1) & (p >= 0.5))[0]
    idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
    X, y = X_test[idx].copy(), y_test[idx].copy()
    r, pr = compute(ca, model, X, y, lower, upper, names)
    r.to_csv(abl.OUTPUT_DIR / "paired_tests.csv", index=False); pr.to_csv(abl.OUTPUT_DIR / "perturbation_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print("\nPAIRED TESTS (conventional vs constraint-aware)")
    print(r.to_string(index=False))
    print("\nPERTURBATION SIZE")
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()