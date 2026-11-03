from __future__ import annotations

import numpy as np
import pytest

from parpi.config import BaselineConfig, FeatureConfig, QCConfig, SplitConfig
from parpi.data.datasets import resistance_target
from parpi.evaluation import (
    classification_metrics,
    expected_calibration_error,
    external_metrics,
    make_folds,
)
from parpi.models.base import Ensemble, Inputs, Targets
from parpi.models.baselines import HGBModel, RidgeModel
from parpi.models.dose_response import DoseResponseModel, leave_one_dose_out
from parpi.preprocess import FeatureSpace

QC = QCConfig(min_genes=50, min_counts=300, max_counts=10**6, max_pct_mito=20, min_cells_per_gene=5)


@pytest.fixture(scope="module")
def xy(synthetic):
    cd, gs = synthetic
    fs = FeatureSpace.fit(cd, gs, QC, FeatureConfig(gene_sets="synthetic"))
    _, ranks, _ = fs.transform(cd, apply_qc=False)
    from parpi.data.synthetic import DOSES

    dose = cd.obs["sample"].map(DOSES).to_numpy(float)
    state = np.where(dose >= 80, "late", np.where(dose >= 10, "mid", "early"))
    x = Inputs(ranks, fs.pathway_scores(ranks))
    return x, Targets(resistance_target(dose, 320), state), dose


def test_leave_dose_out_folds_partition_cells_by_dose(xy):
    _, _, dose = xy
    folds = make_folds(dose, SplitConfig(), seed=0)
    allidx = np.concatenate(folds)
    assert sorted(allidx.tolist()) == list(range(len(dose)))
    for f in folds:  # a dose never appears in both train and test
        rest = np.setdiff1d(np.arange(len(dose)), f)
        assert not set(dose[f]) & set(dose[rest])


@pytest.mark.parametrize("cls", [RidgeModel, HGBModel])
def test_baselines_learn_planted_signal(cls, xy):
    x, y, dose = xy
    folds = make_folds(dose, SplitConfig(), seed=0)
    test = folds[2]
    train = np.setdiff1d(np.arange(len(x)), test)
    m = cls(BaselineConfig(hgb_max_iter=100)).fit(x.take(train), y.take(train), seed=0)
    pred = m.predict(x.take(test))
    assert pred.score.shape == (len(test),)
    assert np.corrcoef(pred.score, y.score[test])[0, 1] > 0.7
    imp = m.pathway_importance()
    assert imp.shape == (x.pathways.shape[1],) and (imp >= 0).all()
    # Planted resistance programme lives in SET_00..SET_03.
    assert set(np.argsort(-imp)[:4]) & {0, 1, 2, 3}


def test_ensemble_gives_uncertainty_and_state_probs(xy):
    x, y, _ = xy
    ens = Ensemble(lambda: RidgeModel(BaselineConfig()), size=3, name="ridge").fit(x, y, seed=1)
    p = ens.predict(x)
    assert (p.score_std > 0).any()
    assert p.state_proba is not None
    np.testing.assert_allclose(p.state_proba.sum(1), 1, atol=1e-6)
    m = classification_metrics(p.state_proba, p.state_classes, y.state)
    assert m["state_balanced_accuracy"] > m["state_chance_balanced_accuracy"]


def test_ensemble_is_deterministic(xy):
    x, y, _ = xy
    a = Ensemble(lambda: HGBModel(BaselineConfig(hgb_max_iter=50)), 2, "hgb").fit(x, y, seed=5)
    b = Ensemble(lambda: HGBModel(BaselineConfig(hgb_max_iter=50)), 2, "hgb").fit(x, y, seed=5)
    np.testing.assert_array_equal(a.predict(x).score, b.predict(x).score)


def test_dose_response_exact_at_measured_doses_and_flags_extrapolation(xy):
    x, _, dose = xy
    m = DoseResponseModel(n_boot=10).fit(dose, x.pathways)
    pred, extrap = m.predict(np.array([40.0, 640.0]))
    np.testing.assert_allclose(pred[0], x.pathways[dose == 40].mean(0), atol=1e-6)
    assert extrap.tolist() == [False, True]
    lo, hi = m.predict_interval(np.array([30.0]))
    assert (lo <= hi).all()


def test_dose_response_beats_nearest_dose_on_planted_transition(xy):
    x, _, dose = xy
    bench = leave_one_dose_out(dose, x.pathways)
    assert bench["model_mae"].mean() < bench["nearest_dose_mae"].mean()


def test_ece_and_external_metrics():
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert expected_calibration_error(proba, np.array([0, 1])) == pytest.approx(0.15)
    m = external_metrics(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    assert m["auroc"] == 1.0


def test_unlabelled_states_are_ignored(xy):
    x, y, _ = xy
    state = y.state.copy()
    state[: len(state) // 3] = ""  # e.g. control cells without a published state
    m = RidgeModel(BaselineConfig()).fit(x, Targets(y.score, state), seed=0)
    p = m.predict(x)
    assert "" not in set(p.state_classes)
    cm = classification_metrics(p.state_proba, p.state_classes, state)
    assert 0 <= cm["state_balanced_accuracy"] <= 1


def test_leave_dataset_out_folds():
    groups = np.array(["a", "a", "b", "c", "b", "c"])
    folds = make_folds(groups, SplitConfig(strategy="leave_dataset_out"), seed=0)
    assert [sorted(set(groups[f])) for f in folds] == [["a"], ["b"], ["c"]]
    with pytest.raises(ValueError, match="two training datasets"):
        make_folds(np.array(["a", "a"]), SplitConfig(strategy="leave_dataset_out"), seed=0)


def test_state_metrics_skip_folds_without_labels():
    proba = np.array([[0.7, 0.3], [0.4, 0.6]])
    classes = np.array(["S1", "S2"])
    assert classification_metrics(proba, classes, np.array(["", ""])) == {}
