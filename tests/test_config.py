from __future__ import annotations

import pytest
from pydantic import ValidationError

from parpi.config import Config, DatasetConfig, load_config
from tests.conftest import ROOT


def test_default_and_smoke_configs_parse():
    for name in ("default.yaml", "smoke.yaml", "kuramochi.yaml", "ovarian.yaml", "release.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.train_datasets


def test_default_config_has_breast_and_ovarian_external_sets():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    kinds = {d.cancer_type for d in cfg.external_datasets}
    assert {"ovarian", "breast"} <= kinds


def test_fingerprint_is_stable_and_sensitive():
    a = load_config(ROOT / "configs" / "smoke.yaml")
    b = load_config(ROOT / "configs" / "smoke.yaml")
    assert a.fingerprint() == b.fingerprint()
    c = a.model_copy(update={"project": "other"})
    assert c.fingerprint() != a.fingerprint()


def test_requires_training_dataset():
    with pytest.raises(ValidationError):
        Config(datasets=[DatasetConfig(name="x", role="external")])


def test_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        DatasetConfig(name="x", not_a_field=1)


def test_ovarian_config_trains_like_kuramochi():
    a = load_config(ROOT / "configs" / "kuramochi.yaml")
    b = load_config(ROOT / "configs" / "ovarian.yaml")
    assert a.train == b.train and a.features == b.features and a.qc == b.qc
    assert a.train_datasets == b.train_datasets
    assert [d.name for d in b.external_datasets if d.gate] == ["pdx_resistance"]


def test_multiline_variants_preregistered():
    a = load_config(ROOT / "configs" / "multiline_a.yaml")
    b = load_config(ROOT / "configs" / "multiline_b.yaml")
    assert [d.name for d in a.train_datasets] == ["kuramochi", "cov362", "ovsaho"]
    # PDX must stay untouched until the single final evaluation.
    for c in (a, b):
        assert not any(d.name.startswith("pdx") for d in c.datasets)
        assert c.split.strategy == "leave_dataset_out"
    # The variants differ only in the excluded gene sets (and their output folders).
    assert a.train == b.train and a.datasets == b.datasets and a.qc == b.qc
    assert not a.features.exclude_gene_sets and b.features.exclude_gene_sets
    assert a.processed_dir != b.processed_dir


def test_final_pdx_config_reuses_variant_a_models():
    a = load_config(ROOT / "configs" / "multiline_a.yaml")
    f = load_config(ROOT / "configs" / "final_pdx.yaml")
    assert f.train == a.train and f.features == a.features and f.qc == a.qc
    assert f.train_datasets == a.train_datasets
    assert f.output_dir == a.output_dir and f.processed_dir == a.processed_dir
    assert [d.name for d in f.external_datasets if d.gate] == ["pdx_resistance"]


def test_release_config_scope():
    r = load_config(ROOT / "configs" / "release.yaml")
    a = load_config(ROOT / "configs" / "multiline_a.yaml")
    # Retrains exactly the model selected in the pre-registered round (variant A, ridge).
    assert r.train.models == ["ridge"] and r.features == a.features
    assert r.train.predict_states is False  # states not validated across lines
    assert r.train_datasets == a.train_datasets and r.split.strategy == "leave_dataset_out"
    assert r.train.ensemble_size == a.train.ensemble_size and r.train.seed == a.train.seed
    gated = [d.name for d in r.external_datasets if r.is_gated(d)]
    assert gated == ["kuramochi_10x"]  # PDX reported, not part of the in-vitro claim
