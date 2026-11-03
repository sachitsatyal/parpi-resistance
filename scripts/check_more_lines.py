"""Inspect layouts of COV362, OVSAHO (GSE247689) and Kuramochi 10x (GSE247688)."""

import pandas as pd

sets = {
    "cov362": (
        "GSE247689",
        "GSE247689_cov362_metadata_g1.tsv.gz",
        "GSE247689_cov362_counts_all_C_to_T40.csv.gz",
    ),
    "ovsaho": (
        "GSE247689",
        "GSE247689_ovsaho_metadata_g1.tsv.gz",
        "GSE247689_ovsaho_counts_all_C_to_T40.csv.gz",
    ),
    "kuramochi_10x": (
        "GSE247688",
        "GSE247688_kuramochi_C_P10_P320_T10_T320_metadata_g1.tsv.gz",
        "GSE247688_kuramochi_counts_all_C_P10_P320_T10_T320.csv.gz",
    ),
}
for name, (acc, meta_file, counts_file) in sets.items():
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/suppl/"
    meta = pd.read_csv(url + meta_file, sep="\t")
    header = pd.read_csv(url + counts_file, nrows=1, header=None).iloc[0].astype(str)
    key = meta.columns[0]
    print("=" * 70, f"\n{name} | metadata {meta.shape} | header fields {len(header)}")
    print(meta.head(3).to_string())
    print("\nunique values per column:\n" + meta.nunique().to_string())
    for c in meta.columns[1:]:
        if meta[c].nunique() <= 20:
            print(f"\n{c}:\n" + meta[c].value_counts().to_string())
    print("\nfirst header fields:", header.head(4).tolist())
    print("metadata cells found in header:", int(meta[key].astype(str).isin(set(header)).sum()))
