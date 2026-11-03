from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from parpi.config import Config, load_config
from parpi.data.synthetic import DOSES, simulate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic():
    cd, gs = simulate(DOSES, cells_per_condition=40, seed=3)
    return cd, gs


@pytest.fixture
def smoke_config(tmp_path: Path) -> Config:
    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    raw["data_dir"] = str(tmp_path / "data")
    raw["output_dir"] = str(tmp_path / "outputs")
    raw["train"]["models"] = ["ridge", "hgb"]
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
