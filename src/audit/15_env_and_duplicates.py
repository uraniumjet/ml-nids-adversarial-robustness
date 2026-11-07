"""
Environment versions + train/test duplicate audit (answers reviewer comments R28, R29, R31).
Run from the project root:
    python src/audit/15_env_and_duplicates.py data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv
"""
import sys, platform, os
import numpy as np, pandas as pd

print("ENVIRONMENT")
print("python", platform.python_version(), "|", platform.platform())
for m in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "joblib"):
    try: print(m, __import__(m).__version__)
    except Exception as e: print(m, "not available")
print("cpu_count", os.cpu_count())
try:
    import subprocess
    print(subprocess.run(["bash", "-c", "lscpu | grep 'Model name'; free -h | head -2; nvidia-smi -L 2>/dev/null || echo 'no GPU found'"],
                         capture_output=True, text=True).stdout)
except Exception: pass

tr, te = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
feat = [c for c in tr.columns if c not in ("id", "attack_cat", "label")]
print("DUPLICATE AUDIT (features only, id/attack_cat/label excluded)")
print("train rows", len(tr), "| test rows", len(te))
print("exact duplicate rows inside train:", int(tr.duplicated(subset=feat).sum()))
print("exact duplicate rows inside test :", int(te.duplicated(subset=feat).sum()))
key = lambda d: d[feat].astype(str).agg("|".join, axis=1)
ktr, kte = key(tr), key(te)
shared = kte.isin(set(ktr))
print("test rows whose feature vector also appears in train:", int(shared.sum()), f"({100*shared.mean():.2f}%)")
if shared.any():
    lab = te.loc[shared, "label"].values
    trmap = tr.assign(k=ktr).groupby("k")["label"].agg(lambda s: s.mode().iat[0])
    same = (trmap.reindex(kte[shared]).values == lab)
    print("   of these, label equals the train label:", int(same.sum()), "| differs:", int((~same).sum()))
    print("   by test label:", pd.Series(lab).value_counts().to_dict())
print("train rows with identical features but conflicting labels:",
      int((tr.assign(k=ktr).groupby("k")["label"].nunique() > 1).sum()))