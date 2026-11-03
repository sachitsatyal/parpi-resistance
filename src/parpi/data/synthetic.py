"""Synthetic dose-escalation data with known ground truth.

Used by tests and CI smoke runs. The generator plants:
  * a *resistance programme* (a few gene sets) that switches on non-linearly
    with log-dose (sharp transition between ~T40 and ~T80),
  * a *dose-stress programme* that rises linearly with log-dose,
  * a repressed programme,
  * nuisance variation in library size that is independent of dose, so a
    model leaning on sequencing depth is caught by the evaluation.
External sets get a different baseline (another "cell line") with parental vs
resistant populations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from parpi.celldata import CellData
from parpi.genesets import GeneSets, parse_gmt

DOSES = {
    "C": 0.0,
    "T1": 1,
    "T2.5": 2.5,
    "T5": 5,
    "T10": 10,
    "T20": 20,
    "T40": 40,
    "T80": 80,
    "T160": 160,
    "T320": 320,
}


def resistance_curve(dose: np.ndarray) -> np.ndarray:
    """Ground-truth resistance (0..1) as a function of µM dose."""
    return 1.0 / (1.0 + np.exp(-(np.log2(dose + 1.0) - np.log2(60.0)) / 0.45))


def make_gene_sets(
    n_genes: int = 600, n_sets: int = 30, set_size: int = 25, seed: int = 0
) -> tuple[np.ndarray, GeneSets]:
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:04d}" for i in range(n_genes)])
    lines = []
    for k in range(n_sets):
        members = rng.choice(genes, size=set_size, replace=False)
        lines.append("\t".join([f"SET_{k:02d}", "", *members]))
    return genes, parse_gmt("\n".join(lines), "synthetic")


def simulate(
    conditions: dict[str, float],
    cells_per_condition: int = 150,
    n_genes: int = 600,
    seed: int = 0,
    line_shift: float = 0.0,
    resistant_override: dict[str, float] | None = None,
) -> tuple[CellData, GeneSets]:
    """Simulate counts for the given ``{condition: dose}`` mapping."""
    genes, gs = make_gene_sets(n_genes=n_genes, seed=0)  # same library for every dataset
    rng = np.random.default_rng(seed)
    gidx = {g: i for i, g in enumerate(genes)}
    sets = [np.array([gidx[g] for g in m]) for m in gs.members]

    base = rng.normal(0.0, 1.0, n_genes)  # log-mean expression per gene
    base += line_shift * rng.normal(0.0, 1.0, n_genes)  # cell-line-specific baseline

    X_parts, obs_parts = [], []
    for cond, dose in conditions.items():
        r = (resistant_override or {}).get(cond, float(resistance_curve(np.array([dose]))[0]))
        stress = np.log2(dose + 1.0) / np.log2(321.0)
        n = cells_per_condition
        logmu = np.tile(base, (n, 1))
        cell_r = np.clip(r + rng.normal(0, 0.12, n), 0, 1)  # within-dose heterogeneity
        for k, eff in ((0, 1.6), (1, 1.3), (2, 1.0)):
            logmu[:, sets[k]] += eff * cell_r[:, None]
        logmu[:, sets[3]] -= 1.2 * cell_r[:, None]
        logmu[:, sets[4]] += 0.9 * stress
        depth = rng.lognormal(mean=np.log(4000), sigma=0.35, size=n)  # dose-independent
        p = np.exp(logmu)
        p /= p.sum(axis=1, keepdims=True)
        counts = rng.poisson(p * depth[:, None])
        mito = rng.uniform(0.01, 0.08, n)
        X_parts.append(sp.csr_matrix(counts.astype(np.float32)))
        obs_parts.append(
            pd.DataFrame(
                {
                    "barcode": [f"{cond}_{i}" for i in range(n)],
                    "sample": cond,
                    "true_resistance": cell_r,
                    "pct_mito_true": mito * 100,
                }
            )
        )
    # Add mitochondrial genes so QC has something to compute.
    X = sp.vstack(X_parts).tocsr()
    obs = pd.concat(obs_parts, ignore_index=True)
    mt_counts = (obs["pct_mito_true"].to_numpy() / 100 * np.asarray(X.sum(1)).ravel()).round()
    X = sp.hstack([X, sp.csr_matrix(mt_counts[:, None].astype(np.float32))]).tocsr()
    all_genes = np.append(genes, "MT-CO1")
    return CellData(X=X, obs=obs.drop(columns="pct_mito_true"), genes=all_genes), gs


def write_synthetic_gmt(path: Path) -> None:
    _, gs = make_gene_sets()
    gs.to_gmt(path)


def synthetic_dataset(kind: str, seed: int, cells: int) -> CellData:
    if kind == "train":
        cd, _ = simulate(DOSES, cells_per_condition=cells, seed=seed)
    else:
        cd, _ = simulate(
            {"parental": 0.0, "resistant": 320.0},
            cells_per_condition=cells,
            seed=seed,
            line_shift=0.3,
            resistant_override={"parental": 0.0, "resistant": 1.0},
        )
    return cd
