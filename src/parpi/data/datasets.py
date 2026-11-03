"""Turn a :class:`DatasetConfig` into a labelled :class:`CellData`."""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

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
        patterns = cfg.files + ([cfg.metadata] if cfg.metadata else [])
        if cfg.source == "geo":
            if not cfg.accession:
                raise ValueError(f"dataset {cfg.name}: source=geo needs an accession")
            files = fetch_geo(cfg.accession, patterns, raw_dir)
        else:
            files = sorted({p for pat in patterns for p in raw_dir.glob(pat)})
        files = [f for f in files if not f.name.endswith((".tar", "MANIFEST.json"))]
        meta_files = [f for f in files if cfg.metadata and fnmatch.fnmatch(f.name, cfg.metadata)]
        count_files = [f for f in files if f not in meta_files]
        if not count_files:
            raise FileNotFoundError(f"dataset {cfg.name}: no count files found in {raw_dir}")
        parts = [read_any(f) for f in count_files]
        universe = np.array(sorted(set().union(*[set(p.genes) for p in parts])))
        cd = CellData.concat([p.align_genes(universe)[0] for p in parts])
        if cfg.barcode_sample_regex:
            extracted = cd.obs["barcode"].astype(str).str.extract(cfg.barcode_sample_regex)[0]
            cd.obs["sample"] = extracted.fillna(cd.obs["sample"])
        if cfg.metadata:
            if len(meta_files) != 1:
                raise FileNotFoundError(
                    f"dataset {cfg.name}: expected one metadata file matching "
                    f"'{cfg.metadata}', found {[f.name for f in meta_files]}"
                )
            cd = attach_metadata(cd, meta_files[0], cfg)
    return annotate(cd, cfg)


def attach_metadata(cd: CellData, path: Path, cfg: DatasetConfig) -> CellData:
    """Join a per-cell metadata table onto the counts by cell ID; keep matched cells only."""
    sep = "," if ".csv" in path.name.lower() else "\t"
    meta = pd.read_csv(path, sep=sep)
    key = cfg.metadata_cell_column
    if key not in meta:
        raise KeyError(f"{path.name}: no column '{key}'. Columns: {list(meta.columns)}")
    meta = meta.drop_duplicates(key)
    meta.index = meta[key].astype(str)
    ids = cd.obs["barcode"].astype(str)
    matched = ids.isin(meta.index).to_numpy()
    # An ID-format mismatch shows up as metadata cells missing from the counts.
    # (Count matrices often hold many more cells than a curated metadata table.)
    meta_found = float(meta.index.isin(set(ids)).mean()) if len(meta) else 0.0
    log.info(
        "%s: %d / %d cells matched to metadata (%.0f%% of metadata cells found)",
        cfg.name,
        matched.sum(),
        len(matched),
        100 * meta_found,
    )
    if meta_found < 0.5:
        raise ValueError(
            f"{cfg.name}: only {meta_found:.0%} of metadata cells are in the count matrix "
            f"(counts: {ids.head(3).tolist()}, metadata: {meta.index[:3].tolist()})"
        )
    if cfg.metadata_required:
        cd = cd.subset(cells=matched)
    # Left join: cells without metadata get NaN in the metadata columns.
    extra = meta.reindex(cd.obs["barcode"].astype(str)).reset_index(drop=True)
    extra = extra.drop(columns=[c for c in extra.columns if c in cd.obs.columns or c == key])
    cd.obs = pd.concat([cd.obs.reset_index(drop=True), extra], axis=1)
    if cfg.metadata_sample_column:
        meta_sample = cd.obs[cfg.metadata_sample_column]
        if cfg.barcode_sample_regex:
            _check_doses(cd.obs["sample"], meta_sample, cfg)
        else:
            cd.obs["sample"] = meta_sample.astype(str).where(meta_sample.notna(), cd.obs["sample"])
    return cd


def _check_doses(from_barcode: pd.Series, from_meta: pd.Series, cfg: DatasetConfig) -> None:
    """Barcode-derived and metadata-derived labels must give the same dose."""
    a = from_barcode.map(cfg.dose_map)
    b = from_meta.map(cfg.dose_map)
    both = a.notna() & b.notna()
    bad = int((a[both] != b[both]).sum())
    if bad:
        raise ValueError(
            f"{cfg.name}: {bad} cells have a barcode dose that disagrees with metadata"
        )
    log.info("%s: barcode doses agree with metadata for %d cells", cfg.name, int(both.sum()))


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
    if cfg.state_column:
        if cfg.state_column not in obs:
            raise KeyError(f"{cfg.name}: state column '{cfg.state_column}' not found")
        raw_state = obs[cfg.state_column]
        has_state = raw_state.notna() & ~raw_state.astype(str).isin(cfg.state_ignore)
        obs["state"] = raw_state.astype(str).where(has_state)
    cd.obs = obs
    return cd


def resistance_target(dose_um: np.ndarray, max_dose: float) -> np.ndarray:
    """Position along the escalation on a log scale: 0 = drug-naive, 1 = top dose."""
    return np.log2(np.asarray(dose_um, dtype=float) + 1.0) / np.log2(max_dose + 1.0)
