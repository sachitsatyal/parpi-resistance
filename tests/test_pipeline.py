"""End-to-end: synthetic data with planted signal must be recovered, reproducibly."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
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

    # In-vitro release gate (leave-dataset-out) + scope-aware model card.
    from parpi.pipeline import quality_gate
    from parpi.report import write_model_card

    evaluate_external(cfg)
    assert quality_gate(cfg) == []
    assert any("AUROC control vs top" in f for f in quality_gate(cfg, min_line_auroc=1.01))
    card = write_model_card(cfg).read_text(encoding="utf-8")
    assert "Validated for" in card and "held-out line" in card
    assert "not a tumor or patient" in card

    # gated_datasets overrides the dataset's own flag; unknown names are refused.
    raw["gated_datasets"] = []
    path.write_text(yaml.safe_dump(raw))
    ungated = load_config(path)
    assert not ungated.is_gated(ungated.external_datasets[0])
    raw["gated_datasets"] = ["nope"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="gated_datasets"):
        load_config(path)


def test_top_pathways_are_signed_for_linear_models(smoke_config):
    """Resistant cells: SET_00-02 high and SET_03 low, all four raising the score."""
    from collections import Counter

    prepare(smoke_config)
    train(smoke_config)
    bundle = ModelBundle.load(paths(smoke_config)["models"] / "ridge")
    assert bundle._signed_weights() is not None
    cd, _ = simulate(
        {"resistant": 320.0}, cells_per_condition=60, seed=12, resistant_override={"resistant": 1.0}
    )
    tops = Counter(p for t in bundle.predict(cd, top_k=3)["top_pathways"] for p in t.split(";"))
    up = "(high, raises score)"
    planted = sum(tops[f"SET_0{i} {up}"] for i in range(3)) + tops["SET_03 (low, raises score)"]
    assert planted > 0.5 * sum(tops.values())
    assert tops["SET_03 (high, raises score)"] == 0  # its weight is negative


def test_untreated_cells_explained_by_what_lowers_the_score(smoke_config):
    from collections import Counter

    prepare(smoke_config)
    train(smoke_config)
    bundle = ModelBundle.load(paths(smoke_config)["models"] / "ridge")
    cd, _ = simulate({"control": 0.0}, cells_per_condition=60, seed=13)
    tags = Counter(
        p.rsplit("(", 1)[1]
        for t in bundle.predict(cd, top_k=3)["top_pathways"]
        for p in t.split(";")
    )
    assert sum(v for k, v in tags.items() if "lowers" in k) > sum(tags.values()) / 2


def test_clipped_cells_keep_their_uncertainty():
    from parpi.models.base import Ensemble, Prediction

    class Fixed:
        name = "fixed"

        def __init__(self, raw):
            self.raw = np.array(raw)

        def predict(self, x):
            return Prediction(np.clip(self.raw, 0, 1), np.zeros(1), raw_score=self.raw)

    ens = Ensemble(lambda: None, 2, "e")  # type: ignore[arg-type,return-value]
    ens.members = [Fixed([-0.2]), Fixed([-0.1])]
    pred = ens.predict(None)  # type: ignore[arg-type]
    assert pred.score[0] == 0.0
    assert pred.score_std[0] == pytest.approx(0.05)


def test_predict_states_off_drops_state_outputs(smoke_config, tmp_path):
    import yaml

    from parpi.config import load_config
    from tests.conftest import ROOT

    raw = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text())
    raw.update(data_dir=str(tmp_path / "d"), output_dir=str(tmp_path / "o"))
    raw["train"].update(models=["ridge"], predict_states=False)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    from parpi.data import datasets as ds_mod

    orig = ds_mod.annotate

    def with_states(cd, dcfg):
        out = orig(cd, dcfg)
        out.obs["state"] = np.where(out.obs.get("dose_um", 0) >= 80, "late", "early")
        return out

    mp = pytest.MonkeyPatch()
    mp.setattr(ds_mod, "annotate", with_states)
    try:
        prepare(cfg)
    finally:
        mp.undo()
    train(cfg)
    cd, _ = simulate({"x": 40.0}, cells_per_condition=10, seed=3)
    pred = ModelBundle.load(paths(cfg)["models"] / "ridge").predict(cd)
    assert "predicted_state" not in pred.columns


def test_explanations_skip_reactome_names(smoke_config):
    """Sets named like Reactome (``... R-HSA-<id>``) never appear unless explain='all'."""
    from dataclasses import replace

    prepare(smoke_config)
    train(smoke_config)
    bundle = ModelBundle.load(paths(smoke_config)["models"] / "ridge")
    gs = bundle.features.gene_sets
    renamed = tuple(f"{n} R-HSA-{i}" if i < 2 else n for i, n in enumerate(gs.names))
    bundle.features.gene_sets = replace(gs, names=renamed)
    cd, _ = simulate(
        {"resistant": 320.0}, cells_per_condition=30, seed=14, resistant_override={"resistant": 1.0}
    )
    shown = {p for t in bundle.predict(cd, top_k=3)["top_pathways"] for p in t.split(";")}
    assert not any("R-HSA-" in p for p in shown)
    shown_all = {
        p for t in bundle.predict(cd, top_k=3, explain="all")["top_pathways"] for p in t.split(";")
    }
    assert any("R-HSA-" in p for p in shown_all)
    with pytest.raises(ValueError, match="explain"):
        bundle.predict(cd, explain="kegg")
