"""
Mask sensitivity + integer-feasibility check (answers reviewer comments R37 and R38).

Part 1  Re-run the constraint-aware attack under three mutable-feature definitions:
   paper27      your current 27 features
   no_derived21 paper27 minus features that are arithmetic functions of others
                (rate, sload, dload, smean, dmean, tcprtt)
   sender9      only features set by the sending host's own packets
                (dur, spkts, sbytes, sttl, sinpkt, sjit, swin, stcpb, trans_depth)
   each with and without clipping to training bounds.
Part 2  Feasibility post-processing: take the clipped adversarial examples back to raw
   units, round integer-valued features, clip to the raw training range, re-standardise,
   and recompute ASR.  (Does the attack survive integrality?)

Place next to 08_constraint_ablation.py and run:  python src/attacks/09_mask_sensitivity.py
"""
from pathlib import Path
import importlib.util, json
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)

DERIVED = ["rate", "sload", "dload", "smean", "dmean", "tcprtt"]
SENDER = ["dur", "spkts", "sbytes", "sttl", "sinpkt", "sjit", "swin", "stcpb", "trans_depth"]
INTEGER = ["spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
           "swin", "dwin", "stcpb", "dtcpb", "trans_depth", "response_body_len"]


def make_masks(names):
    _, mut27, _ = abl.build_masks(names)
    short = [n.replace("numerical__", "") for n in names]
    def by_list(lst):
        m = np.zeros(len(names), bool)
        for f in lst:
            assert f in short, f"feature {f} not found"
            m[short.index(f)] = True
        return m
    return {"paper27": mut27,
            "no_derived21": mut27 & ~by_list(DERIVED),
            "sender9": by_list(SENDER)}, short


def get_numeric_scaler(preprocessor):
    t = preprocessor.named_transformers_["numerical"]
    return t.steps[-1][1] if hasattr(t, "steps") else t


def make_feasible(scaler, Xa, short, lower, upper):
    """Round integer features and clip to raw training range, in raw units."""
    Xo = Xa.copy(); n = len(scaler.mean_)
    raw = scaler.inverse_transform(Xa[:, :n])
    lo_raw = scaler.inverse_transform(lower[None, :n])[0]
    hi_raw = scaler.inverse_transform(upper[None, :n])[0]
    for f in INTEGER:
        j = short.index(f); raw[:, j] = np.rint(raw[:, j])
    raw = np.minimum(np.maximum(raw, lo_raw), hi_raw)
    raw = np.maximum(raw, 0.0)
    Xo[:, :n] = scaler.transform(raw)
    return Xo


def run(ca, model, X_attack, y_attack, lower, upper, names, scaler=None, epsilons=abl.EPSILONS):
    masks, short = make_masks(names)
    rows = []
    for mname, mask in masks.items():
        for clip in (False, True):
            for kind in ("FGSM", "PGD"):
                for eps in epsilons:
                    Xa = abl.attack_once(ca, model, X_attack, y_attack, eps, kind, mask, lower, upper, clip)
                    r = {"mask": mname, "n_mutable": int(mask.sum()), "clipping": clip,
                         "attack": kind, "epsilon": eps}
                    r.update(abl.metrics(ca, model, X_attack, Xa, y_attack))
                    if scaler is not None and clip:
                        Xf = make_feasible(scaler, Xa, short, lower, upper)
                        r["asr_after_rounding"] = abl.metrics(ca, model, X_attack, Xf, y_attack)["asr"]
                    rows.append(r)
    return pd.DataFrame(rows)


def main():
    abl.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
    X_train = pd.read_csv(abl.X_TRAIN_PATH).values
    X_test = pd.read_csv(abl.X_TEST_PATH).values
    y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
    model = joblib.load(abl.MODEL_PATH)
    meta = json.load(open(abl.META_PATH, encoding="utf-8"))
    prep = joblib.load(abl.PREP_PATH)
    names = abl.build_feature_names(meta, prep, X_test.shape[1])
    lower, upper = X_train.min(axis=0), X_train.max(axis=0)
    p = ca.get_positive_class_probability(model, X_test)
    eligible = np.where((y_test == 1) & ((p >= 0.5).astype(int) == 1))[0]
    idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
    X_attack, y_attack = X_test[idx].copy(), y_test[idx].copy()
    try:
        scaler = get_numeric_scaler(prep)
    except Exception as e:
        print("Could not read the numeric scaler, skipping rounding check:", e); scaler = None
    res = run(ca, model, X_attack, y_attack, lower, upper, names, scaler)
    res.to_csv(abl.OUTPUT_DIR / "mask_sensitivity.csv", index=False)
    print("\nMASK SENSITIVITY at eps=0.20")
    print(res[res.epsilon == 0.20].drop(columns=["epsilon"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()