"""Typed, validated configuration.

Every run is fully described by one YAML file. The resolved config is hashed and
written next to the run's artifacts so any result can be traced to the exact
settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelName = Literal["ridge", "hgb", "hierarchical"]
MODEL_NAMES: tuple[ModelName, ...] = ("ridge", "hgb", "hierarchical")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    """How to obtain and label one dataset."""

    name: str
    role: Literal["train", "external"] = "train"
    cancer_type: Literal["ovarian", "breast", "other"] = "ovarian"
    source: Literal["geo", "local", "synthetic"] = "geo"
    accession: str | None = None
    # Glob(s) selecting supplementary files to download / read.
    files: list[str] = Field(default_factory=list)
    # Local path (source=local) or cache dir override.
    path: str | None = None
    # Regex with one capture group applied to the sample name to extract the
    # condition label (e.g. "C", "T40", "resistant").
    condition_regex: str = r"(.+)"
    # Map condition label -> drug concentration in µM. Control = 0.
    dose_map: dict[str, float] = Field(default_factory=dict)
    # For external sets: map condition label -> binary resistant (1) / parental (0).
    resistant_map: dict[str, int] = Field(default_factory=dict)
    # External sets only: False = scored and reported, but never blocks a release.
    gate: bool = True
    # Optional obs column holding published cell-state labels (e.g. França S1-S5).
    state_column: str | None = None
    # Optional per-cell metadata table (glob, also downloaded) for datasets that ship
    # one combined count matrix: rows are matched to cells by `metadata_cell_column`,
    # and `metadata_sample_column` supplies each cell's sample/condition label.
    metadata: str | None = None
    metadata_cell_column: str = "cell"
    metadata_sample_column: str | None = None
    # True: keep only cells present in the metadata (authors' curated set).
    # False: keep every cell; metadata columns are filled where available.
    metadata_required: bool = True
    # Regex with one group taking each cell's sample label from its barcode
    # (e.g. "bc_40uM_123.456" -> "40uM"). When set together with
    # `metadata_sample_column`, the metadata label is only used to cross-check doses.
    barcode_sample_regex: str | None = None
    # Values of `state_column` to ignore as states (e.g. the control label "C").
    state_ignore: list[str] = Field(default_factory=list)
    # Synthetic generator settings (source=synthetic only).
    synthetic: dict[str, float | int] = Field(default_factory=dict)


class QCConfig(_Strict):
    min_genes: int = 200
    min_counts: int = 500
    max_counts: int = 50_000
    max_pct_mito: float = 20.0
    min_cells_per_gene: int = 10


class FeatureConfig(_Strict):
    gene_sets: str = "enrichr:MSigDB_Hallmark_2020+enrichr:Reactome_2022"
    min_set_size: int = 10
    max_set_size: int = 500
    # Minimum fraction of the model's gene universe present in a query dataset.
    min_gene_coverage: float = 0.6
    # Case-insensitive regexes; gene sets whose name matches any are dropped.
    exclude_gene_sets: list[str] = Field(default_factory=list)


class SplitConfig(_Strict):
    # Hold out whole dose groups: tests interpolation to unseen doses, not
    # memorisation of cells from doses already seen.
    # leave_dataset_out: hold out a whole training dataset (e.g. one cell line) per fold.
    strategy: Literal["leave_dose_out", "leave_dataset_out", "random_cells"] = "leave_dose_out"
    n_folds: int = 5


class BaselineConfig(_Strict):
    ridge_alpha: float = 10.0
    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.05


class HierarchicalConfig(_Strict):
    heads: int = 4
    dim: int = 64
    dropout: float = 0.2
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 256
    patience: int = 8
    recon_weight: float = 0.1
    state_weight: float = 0.5


class TrainConfig(_Strict):
    models: list[ModelName] = Field(default_factory=lambda: list(MODEL_NAMES))
    ensemble_size: int = 5
    seed: int = 42
    # Train the optional cell-state classifier when state labels exist. Off for a release
    # whose state predictions are not validated (e.g. labels from one line only).
    predict_states: bool = True
    baseline: BaselineConfig = BaselineConfig()
    hierarchical: HierarchicalConfig = HierarchicalConfig()


class Config(_Strict):
    project: str = "parpi-resistance"
    data_dir: str = "data"
    output_dir: str = "outputs"
    # Where featurised data is written; default <data_dir>/processed. Set it when
    # several configs with different feature spaces share one data_dir.
    processed_dir: str | None = None
    # External datasets that may block a release. None = use each dataset's `gate:` flag.
    # Lets a release config narrow or change the claim without editing dataset files.
    gated_datasets: list[str] | None = None
    datasets: list[DatasetConfig]
    qc: QCConfig = QCConfig()
    features: FeatureConfig = FeatureConfig()
    split: SplitConfig = SplitConfig()
    train: TrainConfig = TrainConfig()

    @model_validator(mode="after")
    def _one_train_set(self) -> Config:
        if not any(d.role == "train" for d in self.datasets):
            raise ValueError("config needs at least one dataset with role: train")
        names = [d.name for d in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError(f"dataset names must be unique, got {names}")
        if self.gated_datasets is not None:
            ext = {d.name for d in self.external_datasets}
            unknown = sorted(set(self.gated_datasets) - ext)
            if unknown:
                raise ValueError(f"gated_datasets not among external datasets: {unknown}")
        return self

    def is_gated(self, d: DatasetConfig) -> bool:
        """Whether an external dataset may block a release under this config."""
        if self.gated_datasets is not None:
            return d.name in self.gated_datasets
        return d.gate

    @property
    def train_datasets(self) -> list[DatasetConfig]:
        return [d for d in self.datasets if d.role == "train"]

    @property
    def external_datasets(self) -> list[DatasetConfig]:
        return [d for d in self.datasets if d.role == "external"]

    def fingerprint(self) -> str:
        """Stable short hash of the resolved config."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    """Load a YAML config. ``includes:`` lists other YAMLs whose datasets are appended."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    datasets: list[Any] = []
    for inc in raw.pop("includes", []):
        inc_raw = yaml.safe_load((path.parent / inc).read_text(encoding="utf-8")) or {}
        datasets.extend(inc_raw.get("datasets", []))
    datasets.extend(raw.pop("datasets", []))
    return Config(datasets=datasets, **raw)
