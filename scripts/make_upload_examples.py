"""Make test files for the web app's upload box, from data the model never trained on.

    uv run python scripts/make_upload_examples.py

Uses the Kuramochi 10x data (GSE247688, the platform check; not training data) and
skips every cell already in the app's built-in example, so uploads are new cells.
Writes to upload_examples/ (git-ignored):

- kuramochi_10x_all_samples.h5ad: 150 cells from each sample in the file, including the
  ones the built-in example leaves out. Tests multi-sample .h5ad upload.
- <sample>.csv: one untreated and one top-dose sample, 100 cells each, as a plain
  cells x genes CSV. Tests CSV upload; the app names the sample after the file.

Only the release model's genes (plus MT- genes for QC) are kept to keep files small;
within-cell ranks over the model's genes, and so the scores, are unchanged by this.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from parpi.bundle import ModelBundle
from parpi.data.readers import read_any

RAW = Path("data/raw/kuramochi_10x/GSE247688_kuramochi_counts_all_C_P10_P320_T10_T320.csv.gz")
MODEL = Path("outputs/release/models/ridge")
BUILT_IN = Path("app/examples/kuramochi_10x_example.h5ad")
OUT = Path("upload_examples")
N_H5AD, N_CSV = 150, 100

bundle = ModelBundle.load(MODEL)
cd = read_any(RAW)
used = set(read_any(BUILT_IN).obs["barcode"]) if BUILT_IN.exists() else set()
sample = cd.obs["barcode"].str.extract(r"_([^_]+)$")[0].to_numpy()
fresh = ~cd.obs["barcode"].isin(used).to_numpy()
genes = np.where(np.isin(cd.genes, bundle.features.genes) | np.char.startswith(cd.genes, "MT-"))[0]
rng = np.random.default_rng(1)
samples = sorted(pd.unique(sample))
print("samples in the file:", ", ".join(samples))


def pick(label: str, n: int) -> np.ndarray:
    idx = np.where((sample == label) & fresh)[0]
    return np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))


OUT.mkdir(exist_ok=True)

# 1. Every sample in one .h5ad
picks = [(pick(s, N_H5AD), s) for s in samples]
ex = cd.subset(cells=np.concatenate([c for c, _ in picks]), genes=genes)
ex.obs["sample"] = np.concatenate([[s] * len(c) for c, s in picks])
path = OUT / "kuramochi_10x_all_samples.h5ad"
ex.write_h5ad(path)
print(f"{path}: {ex.n_cells} cells, {len(samples)} samples, {path.stat().st_size / 1e6:.1f} MB")

# 2. One untreated and one top-dose sample as plain CSVs (cells x genes)
untreated = next(s for s in samples if s.startswith("C"))
top = next((s for s in samples if s.startswith("T320.A")), [s for s in samples if "320" in s][-1])
for label in (untreated, top):
    sub = cd.subset(cells=pick(label, N_CSV), genes=genes)
    X = sub.X.toarray().astype(np.int32)
    df = pd.DataFrame(X, index=sub.obs["barcode"].to_numpy(), columns=sub.genes)
    path = OUT / f"kuramochi_10x_{label}.csv"
    df.to_csv(path)
    print(f"{path}: {sub.n_cells} cells, {path.stat().st_size / 1e6:.1f} MB")
