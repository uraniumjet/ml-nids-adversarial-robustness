"""
Export the 1,000-row attack cohort (row indices in the official test file) so others can reproduce it.
Run from the project root:   python src/attacks/17_export_cohort.py
Writes results/attacks/cohort_indices.csv (columns: cohort_rank, test_row_index, label).
"""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("abl", HERE / "08_constraint_ablation.py")
abl = importlib.util.module_from_spec(spec); spec.loader.exec_module(abl)

ca = abl.load_attack_module(abl.ATTACK_MODULE_PATH)
X_test = pd.read_csv(abl.X_TEST_PATH).values; y_test = pd.read_csv(abl.Y_TEST_PATH).values.ravel()
model = joblib.load(abl.MODEL_PATH)
p = ca.get_positive_class_probability(model, X_test)
eligible = np.where((y_test == 1) & (p >= 0.5))[0]
idx = np.sort(np.random.default_rng(abl.SEED).choice(eligible, size=abl.N_SAMPLES, replace=False))
out = pd.DataFrame({"cohort_rank": np.arange(len(idx)), "test_row_index": idx, "label": y_test[idx]})
out.to_csv(abl.OUTPUT_DIR / "cohort_indices.csv", index=False)
print("eligible pool:", len(eligible), "| cohort:", len(idx), "| saved to", abl.OUTPUT_DIR / "cohort_indices.csv")
print("first indices:", idx[:8].tolist())
