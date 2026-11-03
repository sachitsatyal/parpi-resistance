"""Compact summary of external evaluation results: python summarize_external.py <metrics dir>."""

import json
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/metrics")
with open(folder / "external.json") as f:
    ext = json.load(f)
print("results from:", folder)
for ds, res in ext.items():
    print("=" * 60)
    print(ds, "| gated:", res.get("gated"))
    for model, m in res.items():
        if isinstance(m, dict):
            auc = m.get("auroc")
            print(f"  {model:13s} AUROC={auc if auc is None else round(auc, 3)}")
            print("    mean score by sample:", m["mean_score_by_condition"])
