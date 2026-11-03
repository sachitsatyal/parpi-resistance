"""Do the pathway shifts along the Kuramochi continuum reappear in the PDX tumors?

For every pathway, compute a standardised effect (difference in means / pooled SD):
  * Kuramochi: late doses (160-320 uM) vs control, and early doses (1-5 uM) vs control
  * PDX: resistant vs vehicle, persister vs vehicle, and a NULL contrast between the
    two vehicle tumors (V2 vs V1), which shows how big tumor-to-tumor noise is.
Then correlate the effects across all pathways. If a PDX contrast correlates with the
Kuramochi effect clearly more than the NULL does, the biology transfers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from parpi.preprocess import FeatureSpace

P = Path("data/processed")
names = np.array(FeatureSpace.load(P / "feature_space").gene_sets.names)


def load(name):
    return np.load(P / name / "pathways.npy"), pd.read_csv(P / name / "obs.csv")


def effect(X, obs, hi, lo):
    a = X[obs["condition"].isin(hi).to_numpy()]
    b = X[obs["condition"].isin(lo).to_numpy()]
    return (a.mean(0) - b.mean(0)) / (np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-9)


Xk, ok = load("train")
late = effect(Xk, ok, ["160uM", "320uM"], ["ctrlI"])
early = effect(Xk, ok, ["1uM", "2.5uM", "5uM"], ["ctrlI"])
Xr, orr = load("pdx_resistance")
Xp, op = load("pdx_persistence")
persisters = [c for c in op["condition"].unique() if str(c).startswith("D")]
pdx = {
    "PDX resistant (R1, R3) vs vehicle": effect(
        Xr, orr, ["R1.1", "R1.2", "R3.1", "R3.2"], ["V1", "V2"]
    ),
    "PDX R3 off-drug (R3.V) vs vehicle": effect(Xr, orr, ["R3.V"], ["V1", "V2"]),
    "PDX persisters (D1-D4) vs vehicle": effect(Xp, op, persisters, ["V1", "V2"]),
    "NULL: vehicle V2 vs V1 (resistance exp.)": effect(Xr, orr, ["V2"], ["V1"]),
    "NULL: vehicle V2 vs V1 (persistence exp.)": effect(Xp, op, ["V2"], ["V1"]),
}
print(f"Spearman correlation across {len(names)} pathways with the Kuramochi effect\n")
print(f"{'PDX contrast':44s} {'late (160-320 vs C)':>20s} {'early (1-5 vs C)':>18s}")
for k, v in pdx.items():
    r_late, r_early = spearmanr(late, v).statistic, spearmanr(early, v).statistic
    print(f"{k:44s} {r_late:>+20.3f} {r_early:>+18.3f}")

top = np.argsort(-np.abs(late))[:15]
table = pd.DataFrame(
    {
        "kuramochi_late": late[top],
        "pdx_resistant": pdx["PDX resistant (R1, R3) vs vehicle"][top],
        "pdx_persister": pdx["PDX persisters (D1-D4) vs vehicle"][top],
        "null_V2_vs_V1": pdx["NULL: vehicle V2 vs V1 (resistance exp.)"][top],
    },
    index=[n[:60] for n in names[top]],
)
print("\nTop 15 Kuramochi late-dose pathways and their PDX effects (standardised):")
print(table.round(2).to_string())
