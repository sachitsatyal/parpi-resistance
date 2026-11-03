"""A minimal, dependency-light single-cell container.

The pipeline works on :class:`CellData` (sparse counts + per-cell metadata +
gene names). ``anndata`` is only used at the I/O boundary, so the core logic
can be tested without heavy dependencies and is easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class CellData:
    X: sp.csr_matrix  # cells x genes, raw counts (or normalized, see `layer`)
    obs: pd.DataFrame  # one row per cell
    genes: np.ndarray  # gene symbols, length = n_genes
    layer: str = "counts"
    uns: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = sp.csr_matrix(self.X)
        self.genes = np.asarray(self.genes, dtype=str)
        if self.X.shape != (len(self.obs), len(self.genes)):
            raise ValueError(
                f"shape mismatch: X={self.X.shape}, obs={len(self.obs)}, genes={len(self.genes)}"
            )
        if len(set(self.genes)) != len(self.genes):
            self._dedupe_genes()

    def _dedupe_genes(self) -> None:
        """Sum counts of duplicated gene symbols (common after Ensembl->symbol mapping)."""
        codes, uniques = pd.factorize(self.genes)
        agg = sp.csr_matrix(
            (np.ones(len(codes)), (np.arange(len(codes)), codes)),
            shape=(len(codes), len(uniques)),
        )
        self.X = sp.csr_matrix(self.X @ agg)
        self.genes = np.asarray(uniques, dtype=str)

    @property
    def n_cells(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.X.shape[1])

    def subset(self, cells: np.ndarray | None = None, genes: np.ndarray | None = None) -> CellData:
        X, obs, g = self.X, self.obs, self.genes
        if cells is not None:
            X, obs = X[cells], obs.iloc[cells] if cells.dtype != bool else obs[cells]
        if genes is not None:
            X, g = X[:, genes], g[genes]
        return CellData(
            X=X, obs=obs.reset_index(drop=True), genes=g, layer=self.layer, uns=dict(self.uns)
        )

    def align_genes(self, universe: np.ndarray) -> tuple[CellData, float]:
        """Reorder/pad columns to ``universe``. Missing genes become all-zero columns.

        Returns the aligned data and the fraction of the universe that was present.
        """
        index = {g: i for i, g in enumerate(self.genes)}
        src = np.array([index.get(g, -1) for g in universe])
        present = src >= 0
        # Select present columns, then place them with a sparse selection matrix.
        place = sp.csr_matrix(
            (
                np.ones(int(present.sum()), dtype=np.float32),
                (np.arange(int(present.sum())), np.where(present)[0]),
            ),
            shape=(int(present.sum()), len(universe)),
        )
        out = sp.csr_matrix(self.X[:, src[present]] @ place)
        aligned = CellData(
            X=out,
            obs=self.obs.copy(),
            genes=np.asarray(universe),
            layer=self.layer,
            uns=dict(self.uns),
        )
        return aligned, float(present.mean())

    @staticmethod
    def concat(parts: list[CellData]) -> CellData:
        genes = parts[0].genes
        for p in parts[1:]:
            if not np.array_equal(p.genes, genes):
                raise ValueError("concat requires identical gene order; call align_genes first")
        return CellData(
            X=sp.vstack([p.X for p in parts]).tocsr(),
            obs=pd.concat([p.obs for p in parts], ignore_index=True),
            genes=genes,
            layer=parts[0].layer,
        )

    # ------------------------------------------------------------------ I/O
    def write_h5ad(self, path: str | Path) -> None:
        import anndata as ad  # optional dependency

        adata = ad.AnnData(
            X=self.X,
            obs=self.obs.astype({c: str for c in self.obs.columns if self.obs[c].dtype == object}),
        )
        adata.var_names = pd.Index(self.genes)
        adata.write_h5ad(path)

    def write_npz(self, path: str | Path) -> None:
        """Dependency-free on-disk format used for pipeline intermediates."""
        path = Path(path)
        sp.save_npz(path.with_suffix(".X.npz"), self.X)
        self.obs.to_parquet(
            path.with_suffix(".obs.parquet")
        ) if _has_parquet() else self.obs.to_csv(path.with_suffix(".obs.csv"), index=False)
        np.savetxt(path.with_suffix(".genes.txt"), self.genes, fmt="%s")

    @staticmethod
    def read_npz(path: str | Path) -> CellData:
        path = Path(path)
        X = sp.load_npz(path.with_suffix(".X.npz"))
        pq = path.with_suffix(".obs.parquet")
        obs = pd.read_parquet(pq) if pq.exists() else pd.read_csv(path.with_suffix(".obs.csv"))
        genes = np.loadtxt(path.with_suffix(".genes.txt"), dtype=str, ndmin=1)
        return CellData(X=X, obs=obs, genes=genes)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True
