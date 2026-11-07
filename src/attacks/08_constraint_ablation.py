"""
Ablation of the constraint-aware attack (answers reviewer comment R26).

Question: is the drop in attack success caused by (a) freezing 12 features,
(b) clipping to training bounds, or (c) just having fewer mutable features?

Runs (same model, same 1,000 samples, same epsilons, same seed):
  A  all 39 numerical, no clipping            -> your conventional attack
  B  all 39 numerical, clipping               -> effect of bounds alone
  C  your 27 mutable, no clipping             -> effect of freezing 12 alone
  D  your 27 mutable, clipping                -> your constraint-aware attack
  E  random 27-of-39 (N_RANDOM draws)         -> control: does ANY 27 do this?
  F  your 27 + ONE of the 12 fixed features   -> which fixed feature matters
Also: gradient importance of every numerical feature.

Put this file in src/attacks/ next to 06_conventional_attacks.py.
"""
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
X_TRAIN_PATH = ROOT / "data/processed/unsw_X_train.csv"
X_TEST_PATH = ROOT / "data/processed/unsw_X_test.csv"
Y_TEST_PATH = ROOT / "data/processed/unsw_y_test.csv"
MODEL_PATH = ROOT / "results/models/mlp_clean.joblib"
META_PATH = ROOT / "results/logs/unsw_preprocessing_metadata.json"
PREP_PATH = ROOT / "data/processed/unsw_preprocessor.joblib"
ATTACK_MODULE_PATH = ROOT / "src/attacks/06_conventional_attacks.py"
OUTPUT_DIR = ROOT / "results/attacks"

SEED = 42
N_SAMPLES = 1000
EPSILONS = [0.05, 0.10, 0.20]
PGD_STEPS = 10
N_RANDOM = 20          # random 27-of-39 draws


def pgd_step_size(eps):
    # !!! MUST MATCH the step size used in your conventional / constraint-aware
    # scripts, otherwise run A / D will not reproduce 41.5% / 13.6%.
    return 2.5 * eps / PGD_STEPS


def load_attack_module(path):
    spec = importlib.util.spec_from_file_location("conventional_attacks", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_feature_names(meta, preprocessor, n_cols):
    names = [f"numerical__{n}" for n in meta["numerical_features"]]
    enc = preprocessor.named_transformers_["categorical"]
    names += [f"categorical__{n}" for n in enc.get_feature_names_out(meta["categorical_features"])]
    assert len(names) == n_cols, (len(names), n_cols)
    return names


def build_masks(names):
    numerical = np.array([n.startswith("numerical__") for n in names])
    fixed12 = np.array([
        n in {"numerical__is_ftp_login", "numerical__is_sm_ips_ports"}
        or n.startswith("numerical__ct_") for n in names])
    mutable27 = numerical & ~fixed12
    return numerical, mutable27, fixed12


def attack_once(ca, model, X, y, eps, kind, mask, lower, upper, clip):
    if kind == "FGSM":
        Xa = ca.fgsm_attack_mlp(model, X, y, eps, feature_mask=mask)
        if clip:
            Xa = ca.clip_to_feature_bounds(Xa, lower, upper)
            Xa[:, ~mask] = X[:, ~mask]
        return Xa
    return ca.pgd_attack_mlp(
        model, X, y, epsilon=eps, step_size=pgd_step_size(eps), num_steps=PGD_STEPS,
        feature_mask=mask,
        lower_bounds=lower if clip else None, upper_bounds=upper if clip else None)


def metrics(ca, model, X, Xa, y):
    p0 = ca.get_positive_class_probability(model, X)
    p1 = ca.get_positive_class_probability(model, Xa)
    asr = float(np.mean(p1 < 0.5))                    # cohort is all correctly detected malicious
    return {"asr": asr,
            "mean_conf_drop": float(np.mean(p0 - p1)),
            "mean_changed_features": float(np.mean(np.sum(np.abs(Xa - X) > 1e-12, axis=1)))}


def run_ablation(ca, model, X_attack, y_attack, lower, upper, names, seed=SEED,
                 n_random=N_RANDOM, epsilons=EPSILONS):
    numerical, mutable27, fixed12 = build_masks(names)
    rng = np.random.default_rng(seed)
    num_idx = np.flatnonzero(numerical)
    configs = [("A_all39_noclip", numerical, False, "A"),
               ("B_all39_clip", numerical, True, "B"),
               ("C_mut27_noclip", mutable27, False, "C"),
               ("D_mut27_clip", mutable27, True, "D")]
    for j in range(n_random):
        m = np.zeros(len(names), bool); m[rng.choice(num_idx, 27, replace=False)] = True
        configs.append((f"E_random27_{j:02d}_noclip", m, False, "E"))
    for i in np.flatnonzero(fixed12):
        m = mutable27.copy(); m[i] = True
        configs.append((f"F_mut27+{names[i].replace('numerical__','')}_clip", m, True, "F"))
    rows = []
    for label, mask, clip, grp in configs:
        for kind in ("FGSM", "PGD"):
            for eps in epsilons:
                Xa = attack_once(ca, model, X_attack, y_attack, eps, kind, mask, lower, upper, clip)
                r = {"config": label, "group": grp, "attack": kind, "epsilon": eps,
                     "n_mutable": int(mask.sum()), "clipping": clip}
                r.update(metrics(ca, model, X_attack, Xa, y_attack)); rows.append(r)
    return pd.DataFrame(rows)


def gradient_importance(ca, model, X_attack, y_attack, names):
    g = np.abs(ca.mlp_input_gradient(model, X_attack, y_attack)).mean(axis=0)
    numerical, mutable27, fixed12 = build_masks(names)
    df = pd.DataFrame({"feature": [n.replace("numerical__", "") for n in names],
                       "mean_abs_gradient": g, "numerical": numerical,
                       "group": np.where(fixed12, "fixed12", np.where(mutable27, "mutable27", "categorical"))})
    df = df[df.numerical].sort_values("mean_abs_gradient", ascending=False)
    df["share_of_numerical_gradient"] = df.mean_abs_gradient / df.mean_abs_gradient.sum()
    return df.drop(columns="numerical")


def summarise(res):
    s = res.groupby(["group", "attack", "epsilon"]).asr.agg(["mean", "min", "max", "count"]).reset_index()
    return s


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ca = load_attack_module(ATTACK_MODULE_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH).values
    X_test = pd.read_csv(X_TEST_PATH).values
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    model = joblib.load(MODEL_PATH)
    meta = json.load(open(META_PATH, encoding="utf-8"))
    names = build_feature_names(meta, joblib.load(PREP_PATH), X_test.shape[1])
    lower, upper = X_train.min(axis=0), X_train.max(axis=0)

    # identical cohort selection to your constraint-aware script
    p = ca.get_positive_class_probability(model, X_test)
    eligible = np.where((y_test == 1) & ((p >= 0.5).astype(int) == 1))[0]
    idx = np.sort(np.random.default_rng(SEED).choice(eligible, size=N_SAMPLES, replace=False))
    X_attack, y_attack = X_test[idx].copy(), y_test[idx].copy()

    res = run_ablation(ca, model, X_attack, y_attack, lower, upper, names)
    res.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    summarise(res).to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)
    imp = gradient_importance(ca, model, X_attack, y_attack, names)
    imp.to_csv(OUTPUT_DIR / "gradient_importance.csv", index=False)

    # share of cohort values sitting exactly on a training bound (clipping can bind there)
    numerical, mutable27, _ = build_masks(names)
    at_bound = ((X_attack <= lower) | (X_attack >= upper))[:, mutable27].mean(axis=0)
    pd.DataFrame({"feature": np.array(names)[mutable27], "share_at_bound": at_bound}) \
        .sort_values("share_at_bound", ascending=False).to_csv(OUTPUT_DIR / "bound_share.csv", index=False)

    # reproduction check: A and D must match the paper at eps=0.20
    chk = res[(res.epsilon == 0.20) & res.group.isin(["A", "D"])][["config", "attack", "asr"]]
    print("\nREPRODUCTION CHECK (paper: FGSM 0.347 -> 0.088, PGD 0.415 -> 0.136)")
    print(chk.to_string(index=False))
    print("\nRandom 27-of-39 control, eps=0.20 (no clipping):")
    print(res[(res.group == "E") & (res.epsilon == 0.20)].groupby("attack").asr.describe().round(3))
    print("\nGradient share of the 12 fixed features:",
          round(imp[imp.group == "fixed12"].share_of_numerical_gradient.sum(), 3))
    print("\nTop-10 features by mean |gradient|:\n", imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()