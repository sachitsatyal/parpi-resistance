from __future__ import annotations

import pytest
from pydantic import ValidationError

from parpi.config import Config, DatasetConfig, load_config
from tests.conftest import ROOT


def test_default_and_smoke_configs_parse():
    for name in ("default.yaml", "smoke.yaml"):
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
