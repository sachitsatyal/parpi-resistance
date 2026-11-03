"""Readers for the count-matrix formats GEO submitters actually use."""

from __future__ import annotations

import gzip
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from parpi.celldata import CellData

_SUFFIXES = re.compile(
    r"(\.(h5ad|h5|csv|tsv|txt|mtx))?(\.gz)?$|_(filtered_feature_bc_matrix|matrix|counts)$",
    re.IGNORECASE,
)


def sample_name(path: Path) -> str:
    """GSM1234_T40_counts.csv.gz -> T40_counts -> strip GSM prefix and format suffixes."""
    name = path.name
    for _ in range(3):
        name = _SUFFIXES.sub("", name)
    return re.sub(r"^GSM\d+_", "", name)


def read_any(path: str | Path) -> CellData:
    """Dispatch on file type. Returns raw counts with ``obs['sample']`` set."""
    path = Path(path)
    name = path.name.lower()
    if path.is_dir():
        cd = _read_mtx_dir(path)
    elif name.endswith(".h5ad"):
        cd = _read_h5ad(path)
    elif name.endswith(".h5"):
        cd = _read_10x_h5(path)
    elif name.endswith((".mtx", ".mtx.gz")):
        cd = _read_mtx_dir(path.parent, prefix=path.name.split("matrix")[0])
    elif re.search(r"\.(csv|tsv|txt)(\.gz)?$", name):
        cd = _read_dense(path)
    else:
        raise ValueError(f"unsupported file format: {path}")
    if "sample" not in cd.obs:
        cd.obs["sample"] = sample_name(path)
    return cd


def _read_h5ad(path: Path) -> CellData:
    import anndata as ad

    adata = ad.read_h5ad(path)
    X = adata.raw.X if adata.raw is not None else adata.X
    genes = adata.raw.var_names if adata.raw is not None else adata.var_names
    obs = adata.obs.reset_index(names="barcode")
    return CellData(X=sp.csr_matrix(X), obs=obs, genes=np.asarray(genes))


def _read_10x_h5(path: Path) -> CellData:
    import h5py

    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        shape = tuple(grp["shape"][()])  # (genes, cells)
        X = sp.csc_matrix((grp["data"][()], grp["indices"][()], grp["indptr"][()]), shape=shape)
        feats = grp["features"]
        genes = np.array([g.decode() for g in feats["name"][()]])
        barcodes = np.array([b.decode() for b in grp["barcodes"][()]])
    return CellData(X=X.T.tocsr(), obs=pd.DataFrame({"barcode": barcodes}), genes=genes)


def _open(path: Path):  # type: ignore[no-untyped-def]
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def _find(folder: Path, prefix: str, stems: tuple[str, ...]) -> Path:
    for stem in stems:
        for suf in ("", ".gz"):
            p = folder / f"{prefix}{stem}{suf}"
            if p.exists():
                return p
    raise FileNotFoundError(f"none of {stems} found in {folder} with prefix '{prefix}'")


def _read_mtx_dir(folder: Path, prefix: str = "") -> CellData:
    mtx = _find(folder, prefix, ("matrix.mtx",))
    feats = _find(folder, prefix, ("features.tsv", "genes.tsv"))
    bcs = _find(folder, prefix, ("barcodes.tsv",))
    with warnings.catch_warnings():  # scipy>=1.15 announces a sparse-array default change
        warnings.simplefilter("ignore", DeprecationWarning)
        X = sp.csr_matrix(scipy.io.mmread(mtx)).T.tocsr()  # 10x stores genes x cells
    f = pd.read_csv(feats, sep="\t", header=None)
    genes = f.iloc[:, 1 if f.shape[1] > 1 else 0].astype(str).to_numpy()
    barcodes = pd.read_csv(bcs, sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
    cd = CellData(X=X, obs=pd.DataFrame({"barcode": barcodes}), genes=genes)
    if prefix:
        cd.obs["sample"] = re.sub(r"^GSM\d+_", "", prefix.rstrip("_."))
    return cd


def _read_dense(path: Path) -> CellData:
    sep = "," if ".csv" in path.name.lower() else "\t"
    df = pd.read_csv(path, sep=sep, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Orientation heuristic: gene symbols are mostly upper-case alphanumerics and
    # there are usually more genes than cells in a per-sample matrix.
    idx_gene_like = _gene_like_fraction(df.index)
    col_gene_like = _gene_like_fraction(df.columns)
    if idx_gene_like >= col_gene_like:
        df = df.T  # make cells x genes
    return CellData(
        X=sp.csr_matrix(df.to_numpy(dtype=np.float32)),
        obs=pd.DataFrame({"barcode": df.index.astype(str)}),
        genes=df.columns.astype(str).to_numpy(),
    )


def _gene_like_fraction(labels: pd.Index) -> float:
    s = pd.Series(labels.astype(str)[:500])
    return float(s.str.fullmatch(r"[A-Z][A-Z0-9\-\.]{1,14}|ENSG\d+").mean())
