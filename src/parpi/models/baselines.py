"""Strong, simple baselines on pathway scores.

Any more complex model has to beat these on held-out doses and external
datasets to earn its place in the release.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from parpi.config import BaselineConfig
from parpi.models.base import Inputs, Prediction, Targets


class _PathwayModel:
    name = "base"

    def __init__(self, cfg: BaselineConfig) -> None:
        self.cfg = cfg
        self.reg: Pipeline | HistGradientBoostingRegressor
        self.clf: Pipeline | HistGradientBoostingClassifier | None = None
        self._importance: np.ndarray | None = None
        self._imp_data: tuple[np.ndarray, np.ndarray, int]

    def _make(self, seed: int) -> None:
        raise NotImplementedError

    def fit(self, x: Inputs, y: Targets, seed: int) -> _PathwayModel:
        self._make(seed)
        self.reg.fit(x.pathways, y.score)
        labelled = None if y.state is None else y.state != ""
        if (
            labelled is not None and self.clf is not None and len(np.unique(y.state[labelled])) > 1  # type: ignore[index]
        ):
            self.clf.fit(x.pathways[labelled], y.state[labelled])  # type: ignore[index]
        else:
            self.clf = None
        # Importance is computed lazily (only bundles need it, CV does not) on a
        # small held sample, which keeps cross-validation fast.
        idx = np.random.default_rng(seed).choice(len(x), min(len(x), 1000), replace=False)
        self._imp_data = (x.pathways[idx], y.score[idx], seed)
        self._importance = None
        return self

    def _compute_importance(self, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        r = permutation_importance(self.reg, X, y, n_repeats=2, random_state=seed, n_jobs=1)
        return np.clip(r.importances_mean, 0, None)

    def predict(self, x: Inputs) -> Prediction:
        raw = np.asarray(self.reg.predict(x.pathways), dtype=float)
        score = np.clip(raw, 0.0, 1.0)
        proba = classes = None
        if self.clf is not None:
            proba = self.clf.predict_proba(x.pathways)
            classes = np.asarray(self.clf.classes_)
        return Prediction(
            score=score,
            score_std=np.zeros_like(score),
            state_proba=proba,
            state_classes=classes,
            raw_score=raw,
        )

    def pathway_importance(self) -> np.ndarray:
        if self._importance is None:
            X, y, seed = self._imp_data
            self._importance = self._compute_importance(X, y, seed)
        return self._importance


class RidgeModel(_PathwayModel):
    name = "ridge"

    def _make(self, seed: int) -> None:
        self.reg = make_pipeline(StandardScaler(), Ridge(alpha=self.cfg.ridge_alpha))
        self.clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=seed)
        )

    def _compute_importance(self, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        # Standardised coefficients are exact and cheap for a linear model.
        return np.abs(self.reg[-1].coef_)


class HGBModel(_PathwayModel):
    name = "hgb"

    def _make(self, seed: int) -> None:
        kw = dict(
            max_iter=self.cfg.hgb_max_iter,
            learning_rate=self.cfg.hgb_learning_rate,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
        self.reg = HistGradientBoostingRegressor(**kw)
        self.clf = HistGradientBoostingClassifier(**kw)


class DepthOnlyModel:
    """Sanity baseline: predicts from sequencing depth only (should fail).

    If this scores well, labels are confounded with technical covariates.
    """

    name = "depth_only"

    def fit(self, depth: np.ndarray, score: np.ndarray) -> DepthOnlyModel:
        self.reg = Ridge(alpha=1.0).fit(np.log1p(depth)[:, None], score)
        return self

    def predict(self, depth: np.ndarray) -> np.ndarray:
        return np.clip(self.reg.predict(np.log1p(depth)[:, None]), 0, 1)
