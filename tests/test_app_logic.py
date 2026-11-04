from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from parpi.data.synthetic import simulate
from parpi.pipeline import paths, prepare, train
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "app"))
from logic import (
    ALL,
    dose_curve_figure,
    load_bundle,
    page_view,
    sample_choices,
    score_file,
    score_histogram,
)


def test_app_logic_end_to_end(smoke_config, tmp_path):
    prepare(smoke_config)
    train(smoke_config)
    bundle = load_bundle(str(paths(smoke_config)["models"] / "ridge"))
    cd, _ = simulate({"parental": 0.0, "resistant": 320.0}, cells_per_condition=20, seed=4)
    csv = tmp_path / "GSM1_query.csv"
    pd.DataFrame(
        cd.X.toarray(), columns=cd.genes, index=[f"c{i}" for i in range(cd.n_cells)]
    ).to_csv(csv)
    pred, summary, status = score_file(csv, bundle)
    assert len(summary) == 1 and "Gene coverage" in status
    assert score_histogram(pred) is not None
    names = list(bundle.features.gene_sets.names[:2])
    fig = dose_curve_figure(bundle, names)
    assert len(fig.axes[0].lines) == 2
    assert np.isfinite(pred["resistance_score"]).all()


def test_example_h5ad_round_trip(smoke_config, tmp_path):
    """The app's example is an .h5ad written by CellData.write_h5ad; it must score."""
    pytest.importorskip("anndata")
    prepare(smoke_config)
    train(smoke_config)
    bundle = load_bundle(str(paths(smoke_config)["models"] / "ridge"))
    cd, _ = simulate({"parental": 0.0, "resistant": 320.0}, cells_per_condition=20, seed=8)
    path = tmp_path / "example.h5ad"
    cd.write_h5ad(path)
    pred, summary, _ = score_file(path, bundle)
    assert set(summary["sample"]) == {"parental", "resistant"}
    assert np.isfinite(pred["resistance_score"]).all()


def test_page_view_paginates_and_filters():
    pred = pd.DataFrame(
        {
            "barcode": [f"c{i}" for i in range(250)],
            "sample": ["untreated"] * 120 + ["320 uM"] * 130,
            "resistance_score": np.linspace(0, 1, 250, dtype=np.float32),
            "resistance_score_std": np.zeros(250, dtype=np.float32),
        }
    )
    assert sample_choices(pred) == [ALL, "320 uM", "untreated"]
    view, caption, pages = page_view(pred, ALL, 3)
    assert (len(view), pages) == (50, 3)
    assert caption.startswith("Cells 201-250 of 250")
    view, caption, pages = page_view(pred, "320 uM", 1)
    assert set(view["sample"]) == {"320 uM"} and pages == 2 and len(view) == 100
    assert page_view(pred, "320 uM", 99)[1].endswith("(page 2 of 2)")  # clamped
    assert view["resistance_score"].iloc[1] == round(float(pred["resistance_score"].iloc[121]), 3)
