"""Quality control, normalisation and the fitted feature space.

Design choice — per-cell rank features
--------------------------------------
Pathway scores are computed from the *within-cell rank* of each gene (a
singscore-style statistic). They need no dataset-level statistics, so:
  * a user can score a single sample without a matched control,
  * sequencing depth and batch-wide shifts largely cancel out,
  * the transform is identical at training and inference time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from parpi.celldata import CellData
from parpi.config import FeatureConfig, QCConfig
from parpi.genesets import GeneSets, parse_gmt

log = logging.getLogger(__name__)


def qc_metrics(cd: CellData) -> pd.DataFrame:
    X = cd.X
    counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    mt = np.char.startswith(np.char.upper(cd.genes.astype(str)), "MT-")
    mt_counts = np.asarray(X[:, mt].sum(axis=1)).ravel() if mt.any() else np.zeros_like(counts)
    pct_mt = np.divide(
        100 * mt_counts, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0
    )
    return pd.DataFrame({"total_counts": counts, "n_genes": n_genes, "pct_mito": pct_mt})


def filter_cells(cd: CellData, qc: QCConfig) -> CellData:
    m = qc_metrics(cd)
    keep = (
        (m["n_genes"] >= qc.min_genes)
        & (m["total_counts"] >= qc.min_counts)
        & (m["total_counts"] <= qc.max_counts)
        & (m["pct_mito"] <= qc.max_pct_mito)
    ).to_numpy()
    log.info("QC kept %d / %d cells", keep.sum(), len(keep))
    out = cd.subset(cells=keep)
    for col in m.columns:
        out.obs[col] = m.loc[keep, col].to_numpy()
    return out


def normalize_log1p(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    counts = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(target_sum, counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
    Xn = sp.diags(scale.astype(np.float32)) @ X.astype(np.float32)
    Xn = sp.csr_matrix(Xn)
    Xn.data = np.log1p(Xn.data)
    return Xn


def rank_normalize(X: sp.csr_matrix, chunk: int = 2048) -> np.ndarray:
    """Within-cell normalised ranks in [0, 1] (ties averaged), as dense float32."""
    n, g = X.shape
    out = np.empty((n, g), dtype=np.float32)
    for start in range(0, n, chunk):
        block = X[start : start + chunk].toarray()
        out[start : start + chunk] = (rankdata(block, axis=1, method="average") - 0.5) / g
    return out


@dataclass
class FeatureSpace:
    """Everything needed to turn raw counts into model inputs, frozen after fitting."""

    genes: np.ndarray  # gene universe (ordered)
    gene_sets: GeneSets  # filtered to the universe
    qc: QCConfig
    min_gene_coverage: float

    @classmethod
    def fit(
        cls, train: CellData, library: GeneSets, qc: QCConfig, feats: FeatureConfig
    ) -> FeatureSpace:
        detected = np.asarray((train.X > 0).sum(axis=0)).ravel() >= qc.min_cells_per_gene
        expressed = train.genes[detected]
        gs = library.filter(expressed, feats.min_set_size, feats.max_set_size)
        log.info("feature space: %d genes, %d gene sets", len(gs.genes()), len(gs))
        return cls(genes=gs.genes(), gene_sets=gs, qc=qc, min_gene_coverage=feats.min_gene_coverage)

    @property
    def membership(self) -> sp.csr_matrix:
        return self.gene_sets.membership(self.genes)

    def transform(self, cd: CellData, apply_qc: bool = True) -> tuple[CellData, np.ndarray, float]:
        """Return (QC'd data, per-cell gene ranks over the universe, gene coverage)."""
        if apply_qc:
            cd = filter_cells(cd, self.qc)
        aligned, coverage = cd.align_genes(self.genes)
        if coverage < self.min_gene_coverage:
            raise ValueError(
                f"only {coverage:.0%} of the model's {len(self.genes)} genes are present "
                f"(minimum {self.min_gene_coverage:.0%}). Are genes HGNC symbols?"
            )
        # Normalise on *all* genes so depth correction does not depend on the universe.
        norm = normalize_log1p(cd.X)
        norm_cd = CellData(X=norm, obs=cd.obs, genes=cd.genes)
        ranks = rank_normalize(norm_cd.align_genes(self.genes)[0].X)
        return aligned, ranks, coverage

    def pathway_scores(self, ranks: np.ndarray) -> np.ndarray:
        """Mean within-cell rank of each set's genes, centred at 0."""
        M = self.membership
        sizes = np.asarray(M.sum(axis=0)).ravel()
        return np.asarray(ranks @ M) / sizes - 0.5

    # ---------------------------------------------------------- persistence
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.gene_sets.to_gmt(folder / "gene_sets.gmt")
        (folder / "feature_space.json").write_text(
            json.dumps(
                {
                    "genes": self.genes.tolist(),
                    "gene_set_source": self.gene_sets.source,
                    "gene_set_sha256": self.gene_sets.sha256,
                    "qc": self.qc.model_dump(),
                    "min_gene_coverage": self.min_gene_coverage,
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, folder: Path) -> FeatureSpace:
        meta = json.loads((folder / "feature_space.json").read_text())
        gs = parse_gmt((folder / "gene_sets.gmt").read_text(), meta["gene_set_source"])
        return cls(
            genes=np.array(meta["genes"]),
            gene_sets=gs,
            qc=QCConfig(**meta["qc"]),
            min_gene_coverage=meta["min_gene_coverage"],
        )
