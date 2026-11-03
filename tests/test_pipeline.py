"""End-to-end: synthetic data with planted signal must be recovered, reproducibly."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from parpi.bundle import ModelBundle
from parpi.cli import main
from parpi.data.synthetic import simulate
from parpi.pipeline import evaluate_external, paths, prepare, train
from parpi.report import write_model_card


def _cv(cfg):
    out = json.loads((paths(cfg)["metrics"] / "cv.json").read_text())
    for m in out["models"].values():
        m.pop("seconds", None)
    return out


def test_full_pipeline_recovers_signal_and_is_reproducible(smoke_config):
    cfg = smoke_config
    prepare(cfg)
    train(cfg)
    first = _cv(cfg)
    ext = evaluate_external(cfg)
    card = write_model_card(cfg)

    best = first["selected_model"]
    assert first["models"][best]["spearman_dose_means"] > 0.9
    assert first["models"][best]["mae_cell"] < first["models"][best]["mae_predict_train_mean"]
    # Sequencing depth alone carries no signal in the synthetic data -> must not beat the mean.
    depth = first["models"]["depth_only"]
    assert depth["mae_cell"] >= 0.95 * depth["mae_predict_train_mean"]
    assert ext["synthetic_external"][best]["auroc"] > 0.9
    assert "Research use only" in card.read_text()

    # Re-running training from the same prepared data gives identical metrics.
    train(cfg)
    assert _cv(cfg) == first


def test_bundle_predict_and_cli(smoke_config, tmp_path):
    cfg = smoke_config
    prepare(cfg)
    train(cfg)
    model_dir = paths(cfg)["models"] / "ridge"
    bundle = ModelBundle.load(model_dir)

    cd, _ = simulate(
        {"parental": 0.0, "resistant": 320.0},
        cells_per_condition=30,
        seed=11,
        resistant_override={"parental": 0.0, "resistant": 1.0},
    )
    df = bundle.predict(cd)
    assert {"resistance_score", "resistance_score_std", "top_pathways"} <= set(df.columns)
    assert df.groupby("sample")["resistance_score"].mean().diff().iloc[-1] > 0.2
    assert df.attrs["gene_coverage"] > 0.99

    curve = bundle.dose_curve(np.array([5.0, 640.0]))
    assert curve["extrapolated"].tolist() == [False, True]

    # CLI on a CSV counts file (cells x genes).
    csv = tmp_path / "query.csv"
    pd.DataFrame(
        cd.X.toarray(), columns=cd.genes, index=[f"c{i}" for i in range(cd.n_cells)]
    ).to_csv(csv)
    out = tmp_path / "pred.csv"
    res = CliRunner().invoke(main, ["predict", str(csv), "-m", str(model_dir), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert len(pd.read_csv(out)) > 0


def test_quality_gate(smoke_config):
    from parpi.pipeline import quality_gate

    prepare(smoke_config)
    train(smoke_config)
    evaluate_external(smoke_config)
    assert quality_gate(smoke_config) == []
    assert quality_gate(smoke_config, min_spearman=1.01)  # impossible threshold must fail


def test_leave_dataset_out_pipeline(tmp_path):
    """Two synthetic 'cell lines' (different baselines); hold each one out in turn."""
    import yaml

    from parpi.config import load_config
    from tests.conftest import ROOT

    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    second = dict(
        raw["datasets"][0],
        name="synthetic_train_2",
        synthetic={"seed": 5, "cells_per_condition": 40},
    )
    raw["datasets"].insert(1, second)
    raw.update(data_dir=str(tmp_path / "d"), output_dir=str(tmp_path / "o"))
    raw["split"] = {"strategy": "leave_dataset_out"}
    raw["train"]["models"] = ["ridge"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    # Give only the first synthetic line published-style state labels, like Kuramochi.
    from parpi.data import datasets as ds_mod

    orig = ds_mod.annotate

    def annotate_with_states(cd, dcfg):
        out = orig(cd, dcfg)
        if dcfg.name == "synthetic_train":
            out.obs["state"] = np.where(out.obs["dose_um"] >= 80, "late", "early")
        return out

    import pytest

    mp = pytest.MonkeyPatch()
    mp.setattr(ds_mod, "annotate", annotate_with_states)
    try:
        prepare(cfg)
    finally:
        mp.undo()
    train(cfg)
    cv = json.loads((paths(cfg)["metrics"] / "cv.json").read_text())
    assert "per_dataset" in cv["models"]["ridge"]
    path = tmp_path / "c2.yaml"
    raw.update(data_dir=str(tmp_path / "d2"), output_dir=str(tmp_path / "o2"))
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    prepare(cfg)
    train(cfg)
    cv = json.loads((paths(cfg)["metrics"] / "cv.json").read_text())
    ridge = cv["models"]["ridge"]
    assert set(ridge["per_dataset"]) == {"synthetic_train", "synthetic_train_2"}
    assert ridge["spearman_dose_means_by_dataset"] > 0.8
    for m in ridge["per_dataset"].values():
        assert m["auroc_control_vs_320uM"] > 0.9
    summary = json.loads((paths(cfg)["processed"] / "summary.json").read_text())
    assert "synthetic_train_2/T320" in summary["train"]["qc_by_condition"]
