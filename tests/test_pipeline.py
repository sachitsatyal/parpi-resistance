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
