"""Build the web app's one-click example from public data the model never trained on.

    uv run python scripts/make_example.py

Takes 250 untreated (C.R1) and 250 top-dose (T320.A.R1) Kuramochi cells sequenced on 10x
(GSE247688, the platform check; not training data), keeps the release model's genes, and
writes app/examples/kuramochi_10x_example.h5ad (well under 1 MB). Within-cell ranks over the
model's genes are unchanged by dropping other genes, so scores match the full file.
"""

from pathlib import Path

import numpy as np

from parpi.bundle import ModelBundle
from parpi.data.readers import read_any

RAW = Path("data/raw/kuramochi_10x/GSE247688_kuramochi_counts_all_C_P10_P320_T10_T320.csv.gz")
MODEL = Path("outputs/release/models/ridge")
OUT = Path("app/examples/kuramochi_10x_example.h5ad")
N = 250

bundle = ModelBundle.load(MODEL)
cd = read_any(RAW)
sample = cd.obs["barcode"].str.extract(r"_([^_]+)$")[0]
rng = np.random.default_rng(0)
picks = []
for label, name in (("C.R1", "untreated (C.R1)"), ("T320.A.R1", "320 uM on drug (T320.A.R1)")):
    idx = np.where(sample.to_numpy() == label)[0]
    chosen = np.sort(rng.choice(idx, size=min(N, len(idx)), replace=False))
    picks.append((chosen, name))
cells = np.concatenate([c for c, _ in picks])
genes = np.where(np.isin(cd.genes, bundle.features.genes) | np.char.startswith(cd.genes, "MT-"))[0]
ex = cd.subset(cells=cells, genes=genes)
ex.obs["sample"] = np.concatenate([[name] * len(c) for c, name in picks])
OUT.parent.mkdir(parents=True, exist_ok=True)
ex.write_h5ad(OUT)
print(f"{OUT}: {ex.n_cells} cells x {ex.n_genes} genes, {OUT.stat().st_size / 1e6:.2f} MB")
