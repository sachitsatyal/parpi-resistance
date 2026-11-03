"""Inspect GSE247690 PDX layouts: metadata columns, labels, and count-matrix cell IDs."""

import pandas as pd

base = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE247nnn/GSE247690/suppl/GSE247690_pdx_"
for exp in ("resistance", "persistence"):
    meta = pd.read_csv(base + exp + "_metadata_g1.tsv.gz", sep="\t")
    header = pd.read_csv(base + exp + "_counts_all.csv.gz", nrows=1, header=None).iloc[0]
    cells = pd.Series(header.iloc[1:].astype(str).to_numpy())
    key = meta.columns[0]
    print("=" * 70, "\n", exp, "| metadata", meta.shape, "| counts cells", len(cells))
    print(meta.head(3).to_string())
    print("\nunique values per column:\n" + meta.nunique().to_string())
    for c in meta.columns[1:]:
        if meta[c].nunique() <= 15:
            print(f"\n{c}:\n" + meta[c].value_counts().to_string())
    print("\nfirst count-matrix cell IDs:", cells.head(5).tolist())
    print("last count-matrix cell IDs:", cells.tail(5).tolist())
    print(
        "duplicated IDs: counts",
        int(cells.duplicated().sum()),
        "| metadata",
        int(meta[key].duplicated().sum()),
    )
    print("cells in both:", int(cells.isin(set(meta[key].astype(str))).sum()))
