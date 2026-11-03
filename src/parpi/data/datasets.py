"""Turn a :class:`DatasetConfig` into a labelled :class:`CellData`."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from parpi.celldata import CellData
from parpi.config import DatasetConfig
from parpi.data import synthetic
from parpi.data.download import fetch_geo
from parpi.data.readers import read_any

log = logging.getLogger(__name__)


def load_dataset(cfg: DatasetConfig, data_dir: Path) -> CellData:
    raw_dir = Path(cfg.path) if cfg.path else data_dir / "raw" / cfg.name
    if cfg.source == "synthetic":
        cd = synthetic.synthetic_dataset(
            "train" if cfg.role == "train" else "external",
            seed=int(cfg.synthetic.get("seed", 0)),
            cells=int(cfg.synthetic.get("cells_per_condition", 150)),
        )
    else:
        if cfg.source == "geo":
            if not cfg.accession:
                raise ValueError(f"dataset {cfg.name}: source=geo needs an accession")
            files = fetch_geo(cfg.accession, cfg.files, raw_dir)
        else:
            files = sorted(p for pat in cfg.files for p in raw_dir.glob(pat))
        files = [f for f in files if not f.name.endswith((".tar", "MANIFEST.json"))]
        if not files:
            raise FileNotFoundError(f"dataset {cfg.name}: no count files found in {raw_dir}")
        parts = [read_any(f) for f in files]
        universe = np.array(sorted(set().union(*[set(p.genes) for p in parts])))
        cd = CellData.concat([p.align_genes(universe)[0] for p in parts])
    return annotate(cd, cfg)


def annotate(cd: CellData, cfg: DatasetConfig) -> CellData:
    """Attach condition, dose, resistance and state labels from the dataset config."""
    obs = cd.obs
    rx = re.compile(cfg.condition_regex)

    def _cond(sample: str) -> str:
        m = rx.search(str(sample))
        return m.group(1) if m else str(sample)

    obs["dataset"] = cfg.name
    obs["cancer_type"] = cfg.cancer_type
    obs["condition"] = obs["sample"].map(_cond)
    if cfg.dose_map:
        unknown = sorted(set(obs["condition"]) - set(cfg.dose_map))
        if unknown:
            log.warning("%s: dropping cells with unmapped conditions %s", cfg.name, unknown)
            keep = obs["condition"].isin(cfg.dose_map).to_numpy()
            cd = cd.subset(cells=keep)
            obs = cd.obs
        obs["dose_um"] = obs["condition"].map(cfg.dose_map).astype(float)
    if cfg.resistant_map:
        obs["resistant"] = obs["condition"].map(cfg.resistant_map)
    if cfg.state_column and cfg.state_column in obs:
        obs["state"] = obs[cfg.state_column].astype(str)
    cd.obs = obs
    return cd


def resistance_target(dose_um: np.ndarray, max_dose: float) -> np.ndarray:
    """Position along the escalation on a log scale: 0 = drug-naive, 1 = top dose."""
    return np.log2(np.asarray(dose_um, dtype=float) + 1.0) / np.log2(max_dose + 1.0)
