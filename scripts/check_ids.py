"""Diagnose how GSE206125 count-matrix cells line up with the metadata table."""

import pandas as pd

raw = "data/raw/kuramochi/"
cols = pd.read_csv(raw + "GSE206125_kuramochi_all_C_to_T320.csv.gz", nrows=0, index_col=0).columns
meta = pd.read_csv(raw + "GSE206125_kuramochi_C_to_T320_metadata_g1.tsv.gz", sep="\t")
c = pd.Series(cols.astype(str))
m = meta["cell"].astype(str)
in_meta, in_counts = c.isin(set(m)), m.isin(set(c))


def prefix(s):
    return s.str.replace(r"_[^_]*$", "", regex=True)


print("counts cells:", len(c), "| metadata cells:", len(m), "| matched:", int(in_meta.sum()))
print("\nCount-matrix cells by sample prefix (all / with metadata):")
tab = pd.DataFrame({"all": prefix(c).value_counts(), "matched": prefix(c[in_meta]).value_counts()})
print(tab.fillna(0).astype(int).to_string())
print("\nMetadata cells by dose (all / found in counts):")
print(meta.assign(found=in_counts).groupby("orig.ident")["found"].agg(["size", "sum"]).to_string())
print("\nUnmatched metadata IDs (examples):", m[~in_counts].head(6).tolist())
print("\nState by dose:")
print(pd.crosstab(meta["orig.ident"], meta["State"]).to_string())
