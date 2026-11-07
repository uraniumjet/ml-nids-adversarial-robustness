"""
Train/test overlap sensitivity (answers reviewer comments R28 and R35-related concerns).

The duplicate audit found that 8,541 test rows (10.4%) have a feature vector that also appears in the
training partition. This script measures (a) clean performance on test rows with and without a training
twin, and (b) whether the constraint effect on the 1,000-row cohort differs for cohort rows with and
without a training twin.

Run from the project root (raw train file first, raw test file second):
    python src/audit/16_overlap_sensitivity.py data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("abl", ROOT / "src/attacks/08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)


def twin_flags(raw_train, raw_test):
    feat = [c for c in raw_train.columns if c not in ("id", "attack_cat", "label")]
    h = lambda d: pd.util.hash_pandas_object(d[feat], index=False).values
    return np.isin(h(raw_test), h(raw_train))


def clean_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"n": len(y), "accuracy": (tp + tn) / max(len(y), 1), "recall": tp / max(tp + fn, 1),
            "fpr": fp / max(fp + tn, 1), "n_malicious": int((y == 1).sum()), "n_normal": int((y == 0).sum())}


def compute(ca, model, X_test, y_test, twin, lower, upper, names, eps_list=(0.10, 0.20)):
    numerical, mut27, _ = abl.build_masks(names)
    p = ca.get_positive_class_probability(model, X_test); pred = (p >= 0.5).astype(int)
    clean = pd.DataFrame({"full test": clean_metrics(y_test, pred),
                          "rows WITH train twin": clean_metrics(y_test[twin], pred[twin]),
                          "rows WITHOUT train twin": clean_metrics(y_test[~twin], pred[~twin])}).T
    eligible = np.where((y_test == 1) & (p >= 0.5))[0]
    idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
    X, y, tw = X_test[idx].copy(), y_test[idx].copy(), twin[idx]
    rows = []
    for kind in ("FGSM", "PGD"):
        for eps in eps_list:
            Xc = abl.attack_once(ca, model, X, y, eps, kind, numerical, lower, upper, False)
            Xk = abl.attack_once(ca, model, X, y, eps, kind, mut27, lower, upper, True)
            sc = ca.get_positive_class_probability(model, Xc) < 0.5
            sk = ca.get_positive_class_probability(model, Xk) < 0.5
            for lab, m in (("with train twin", tw), ("without train twin", ~tw), ("all cohort", np.ones(len(tw), bool))):
                if m.sum():
                    rows.append({"attack": kind, "epsilon": eps, "group": lab, "n": int(m.sum()),
                                 "asr_conv": sc[m].mean(), "asr_constr": sk[m].mean(),
                                 "rel_reduction": 1 - sk[m].mean() / sc[m].mean() if sc[m].mean() > 0 else np.nan})
    return clean, int(tw.sum()), pd.DataFrame(rows)


def main():
    raw_tr, raw_te = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values
    X_test = pd.read_csv(abl.X_TEST_PATH).values; y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    assert len(raw_te) == len(y_test) and (raw_te["label"].values == y_test).all(), "raw test rows do not match processed test rows"
    twin = twin_flags(raw_tr, raw_te)
    model = joblib.load(abl.MODEL_PATH)
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    names = abl.build_feature_names(meta, joblib.load(abl.PREP_PATH), X_test.shape[1])
    clean, n_tw, res = compute(ca, model, X_test, y_test, twin, X_train.min(0), X_train.max(0), names)
    clean.to_csv(abl.OUTPUT_DIR / "overlap_clean_metrics.csv"); res.to_csv(abl.OUTPUT_DIR / "overlap_asr.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\nTEST ROWS WITH A TRAIN TWIN: {int(twin.sum())} of {len(twin)}")
    print("\nCLEAN METRICS"); print(clean.to_string())
    print(f"\nCOHORT ROWS WITH A TRAIN TWIN: {n_tw} of {abl.N_SAMPLES}")
    print("\nASR BY TWIN STATUS"); print(res.to_string(index=False))


if __name__ == "__main__":
    main()