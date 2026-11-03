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
    # Optional obs column holding published cell-state labels (e.g. França S1-S5).
    state_column: str | None = None
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


class SplitConfig(_Strict):
    # Hold out whole dose groups: tests interpolation to unseen doses, not
    # memorisation of cells from doses already seen.
    strategy: Literal["leave_dose_out", "random_cells"] = "leave_dose_out"
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
    baseline: BaselineConfig = BaselineConfig()
    hierarchical: HierarchicalConfig = HierarchicalConfig()


class Config(_Strict):
    project: str = "parpi-resistance"
    data_dir: str = "data"
    output_dir: str = "outputs"
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
        return self

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
    raw = yaml.safe_load(path.read_text()) or {}
    datasets: list[Any] = []
    for inc in raw.pop("includes", []):
        inc_raw = yaml.safe_load((path.parent / inc).read_text()) or {}
        datasets.extend(inc_raw.get("datasets", []))
    datasets.extend(raw.pop("datasets", []))
    return Config(datasets=datasets, **raw)
