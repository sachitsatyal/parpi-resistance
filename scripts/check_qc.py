"""Compare QC of cells with vs without published metadata (GSE206125)."""

import pandas as pd

obs = pd.read_csv("data/processed/train/obs.csv")
obs["has_meta"] = obs["nUMI"].notna()
cols = ["total_counts", "n_genes", "pct_mito"]
print("Median QC per dose, cells without (False) vs with (True) metadata:")
print(
    obs.groupby(["condition", "has_meta"])[cols].median().round(1).unstack("has_meta").to_string()
)
m = obs[obs["has_meta"]]
print("\nMatched cells, our numbers vs the metadata:")
print("  median ratio total_counts / nUMI:", round((m["total_counts"] / m["nUMI"]).median(), 3))
print("  correlation total_counts vs nUMI:", round(m["total_counts"].corr(m["nUMI"]), 3))
print(
    "  median genes per cell, ours:",
    m["n_genes"].median(),
    "| metadata nGene:",
    m["nGene"].median(),
)
print("\nUnmatched cells, total_counts distribution:")
print(obs.loc[~obs["has_meta"], "total_counts"].describe().round(0).to_string())
