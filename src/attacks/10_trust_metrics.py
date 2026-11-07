"""
Brier / ECE re-analysis (answers reviewer comments R42, R43 and the equations request).

1. Finds which ECE definition (top-label or positive-class, how many bins) reproduces
   your paper's clean ECE = 0.101367 on the full test set.
2. For all 12 attack conditions computes Brier and ECE on THREE evaluation sets:
     full      82,332 test rows, attacked probabilities substituted at the 1,000 cohort rows (your paper)
     balanced  1,000 attacked malicious + 1,000 clean benign rows (50/50), seed 42
     cohort    the 1,000 attacked malicious rows only (Brier and mean P(malicious))
3. Paired bootstrap 95% CIs (percentile) for the change versus clean.

Place next to 08_constraint_ablation.py and run:  python src/attacks/10_trust_metrics.py
"""
from pathlib import Path
import importlib.util, json
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)

TARGET_CLEAN_ECE = 0.101367     # paper, clean full test
TARGET_CLEAN_BRIER = 0.096912
N_BOOT = 1000
BOOT_SEED = 42


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins=15, mode="toplabel"):
    p = np.asarray(p, float); y = np.asarray(y, int)
    if mode == "toplabel":
        conf = np.maximum(p, 1 - p); correct = ((p >= 0.5).astype(int) == y).astype(float)
        b = np.minimum((conf * bins).astype(int), bins - 1)
        n = np.bincount(b, minlength=bins); sc = np.bincount(b, weights=conf, minlength=bins)
        sa = np.bincount(b, weights=correct, minlength=bins)
    else:  # positive-class reliability
        b = np.minimum((p * bins).astype(int), bins - 1)
        n = np.bincount(b, minlength=bins); sc = np.bincount(b, weights=p, minlength=bins)
        sa = np.bincount(b, weights=y.astype(float), minlength=bins)
    return float(np.abs(sa - sc).sum() / len(p))


def find_ece_definition(p, y, target=TARGET_CLEAN_ECE):
    grid = []
    for mode in ("toplabel", "positive"):
        for bins in (10, 15, 20, 25, 30, 50, 100):
            grid.append((abs(ece(p, y, bins, mode) - target), mode, bins, ece(p, y, bins, mode)))
    grid.sort()
    return grid


def boot_delta(fn, p_clean, p_adv, y, B, seed):
    """Paired percentile bootstrap CI for fn(adv) - fn(clean) over rows."""
    rng = np.random.default_rng(seed); n = len(y); d = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        d[i] = fn(p_adv[idx], y[idx]) - fn(p_clean[idx], y[idx])
    return np.percentile(d, [2.5, 97.5])


def build_conditions(ca, model, X_attack, y_attack, lower, upper, names, epsilons=abl.EPSILONS):
    numerical, mut27, _ = abl.build_masks(names)
    out = {}
    for kind in ("FGSM", "PGD"):
        for space, mask, clip in (("conventional", numerical, False), ("constrained", mut27, True)):
            for eps in epsilons:
                Xa = abl.attack_once(ca, model, X_attack, y_attack, eps, kind, mask, lower, upper, clip)
                out[(kind, space, eps)] = ca.get_positive_class_probability(model, Xa)
    return out


def analyse(p_full_clean, y_full, idx, p_coh_clean, conds, benign_idx, ece_bins, ece_mode,
            B=N_BOOT, seed=BOOT_SEED):
    fE = lambda p, y: ece(p, y, ece_bins, ece_mode)
    # balanced set
    yb = np.concatenate([np.ones(len(idx), int), y_full[benign_idx]])
    pb_clean = np.concatenate([p_coh_clean, p_full_clean[benign_idx]])
    rows = []
    for (kind, space, eps), p_adv in conds.items():
        # full
        pf = p_full_clean.copy(); pf[idx] = p_adv
        # balanced
        pb = np.concatenate([p_adv, p_full_clean[benign_idx]])
        for setname, pc, pa, y in (("full", p_full_clean, pf, y_full), ("balanced", pb_clean, pb, yb),
                                   ("cohort", p_coh_clean, p_adv, np.ones(len(idx), int))):
            r = {"set": setname, "attack": kind, "space": space, "epsilon": eps, "n": len(y),
                 "brier_clean": brier(pc, y), "brier_adv": brier(pa, y)}
            r["d_brier"] = r["brier_adv"] - r["brier_clean"]
            lo, hi = boot_delta(brier, pc, pa, y, B, seed); r["d_brier_lo"], r["d_brier_hi"] = lo, hi
            if setname != "cohort":
                r["ece_clean"] = fE(pc, y); r["ece_adv"] = fE(pa, y); r["d_ece"] = r["ece_adv"] - r["ece_clean"]
                lo, hi = boot_delta(fE, pc, pa, y, B, seed); r["d_ece_lo"], r["d_ece_hi"] = lo, hi
            else:
                r["mean_p_clean"] = float(pc.mean()); r["mean_p_adv"] = float(pa.mean())
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values
    X_test = pd.read_csv(abl.X_TEST_PATH).values
    y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    model = joblib.load(abl.MODEL_PATH)
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    names = abl.build_feature_names(meta, joblib.load(abl.PREP_PATH), X_test.shape[1])
    lower, upper = X_train.min(axis=0), X_train.max(axis=0)
    p_full = ca.get_positive_class_probability(model, X_test)
    eligible = np.where((y_test == 1) & (p_full >= 0.5))[0]
    idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
    X_attack, y_attack = X_test[idx].copy(), y_test[idx].copy()
    benign = np.flatnonzero(y_test == 0)
    benign_idx = np.sort(np.random.default_rng(abl.SEED + 1).choice(benign, size=abl.N_SAMPLES, replace=False))

    print("Clean full-test Brier:", round(brier(p_full, y_test), 6), "(paper 0.096912)")
    grid = find_ece_definition(p_full, y_test)
    print("\nECE definition search (closest to paper 0.101367 first):")
    for g in grid[:5]: print(f"  mode={g[1]:9s} bins={g[2]:3d}  ECE={g[3]:.6f}  |diff|={g[0]:.6f}")
    best = grid[0]
    if best[0] > 5e-6: print("  WARNING: no definition reproduces the paper's clean ECE exactly; using the closest.")
    mode, bins = best[1], best[2]
    print(f"Using ECE mode={mode}, bins={bins}")

    conds = build_conditions(ca, model, X_attack, y_attack, lower, upper, names)
    p_coh_clean = p_full[idx]
    res = analyse(p_full, y_test, idx, p_coh_clean, conds, benign_idx, bins, mode)
    res.to_csv(abl.OUTPUT_DIR / "trust_metrics.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nREPRODUCTION (paper Table 5, PGD conventional 0.20: Brier 0.101017, ECE 0.105016)")
    print(res[(res.set == "full") & (res.attack == "PGD") & (res.space == "conventional") & (res.epsilon == 0.20)]
          [["brier_adv", "ece_adv"]].round(6).to_string(index=False))
    cols = ["set", "attack", "space", "n", "d_brier", "d_brier_lo", "d_brier_hi", "d_ece", "d_ece_lo", "d_ece_hi"]
    print("\nCHANGE VERSUS CLEAN at eps=0.20 (95% paired bootstrap CI)")
    print(res[res.epsilon == 0.20][cols].round(5).to_string(index=False))


if __name__ == "__main__":
    main()