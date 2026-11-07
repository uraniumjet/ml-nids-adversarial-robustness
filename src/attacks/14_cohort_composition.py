"""
Cohort composition and ASR by attack category (answers reviewer comment R35).

Reports: size of the eligible pool (malicious test rows the clean MLP detects), how the 1,000-row
cohort was drawn (uniform random without replacement, seed 42), the attack_cat composition of the
pool and of the cohort, and ASR per category for conventional vs constraint-aware FGSM/PGD.

Needs the RAW UNSW-NB15 testing file (it holds attack_cat). Run from the project root:
    python src/attacks/14_cohort_composition.py path/to/UNSW_NB15_testing-set.csv
"""
import sys, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)


def compute(ca, model, X_test, y_test, cats, lower, upper, names, eps_list=(0.10, 0.20)):
    numerical, mut27, _ = abl.build_masks(names)
    p = ca.get_positive_class_probability(model, X_test)
    eligible = np.where((y_test == 1) & (p >= 0.5))[0]
    idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
    X, y = X_test[idx].copy(), y_test[idx].copy(); c = cats[idx]
    pool = pd.Series(cats[eligible]).value_counts(); coh = pd.Series(c).value_counts()
    comp = pd.DataFrame({"pool_n": pool, "pool_pct": 100 * pool / pool.sum(),
                         "cohort_n": coh, "cohort_pct": 100 * coh / coh.sum()}).fillna(0).sort_values("pool_n", ascending=False)
    rows = []
    for kind in ("FGSM", "PGD"):
        for eps in eps_list:
            Xc = abl.attack_once(ca, model, X, y, eps, kind, numerical, lower, upper, False)
            Xk = abl.attack_once(ca, model, X, y, eps, kind, mut27, lower, upper, True)
            sc = ca.get_positive_class_probability(model, Xc) < 0.5
            sk = ca.get_positive_class_probability(model, Xk) < 0.5
            for cat in comp.index:
                m = c == cat
                if m.sum():
                    rows.append({"attack": kind, "epsilon": eps, "attack_cat": cat, "n": int(m.sum()),
                                 "asr_conv": sc[m].mean(), "asr_constr": sk[m].mean()})
    return len(eligible), comp, pd.DataFrame(rows)


def main():
    raw = pd.read_csv(sys.argv[1])
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values
    X_test = pd.read_csv(abl.X_TEST_PATH).values; y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    assert len(raw) == len(y_test), f"raw rows {len(raw)} != processed rows {len(y_test)}"
    assert (raw["label"].values == y_test).all(), "raw labels do not match processed labels: row order differs"
    cats = raw["attack_cat"].astype(str).str.strip().values
    model = joblib.load(abl.MODEL_PATH)
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    names = abl.build_feature_names(meta, joblib.load(abl.PREP_PATH), X_test.shape[1])
    n_pool, comp, byc = compute(ca, model, X_test, y_test, cats, X_train.min(0), X_train.max(0), names)
    comp.to_csv(abl.OUTPUT_DIR / "cohort_composition.csv"); byc.to_csv(abl.OUTPUT_DIR / "asr_by_category.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(f"\nELIGIBLE POOL SIZE: {n_pool}  (cohort drawn uniformly at random without replacement, seed {abl.SEED})")
    print("\nCOMPOSITION"); print(comp.to_string())
    print("\nASR BY CATEGORY"); print(byc.to_string(index=False))


if __name__ == "__main__":
    main()