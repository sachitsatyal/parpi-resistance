"""Common model interface and a bagging ensemble that gives every model uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Inputs:
    ranks: np.ndarray  # cells x genes, within-cell normalised ranks
    pathways: np.ndarray  # cells x gene sets, centred pathway scores

    def __len__(self) -> int:
        return len(self.pathways)

    def take(self, idx: np.ndarray) -> Inputs:
        return Inputs(self.ranks[idx], self.pathways[idx])


@dataclass
class Targets:
    score: np.ndarray  # continuous resistance position in [0, 1]
    # Optional categorical cell-state labels; "" marks a cell with no state label.
    state: np.ndarray | None = None

    def take(self, idx: np.ndarray) -> Targets:
        return Targets(self.score[idx], None if self.state is None else self.state[idx])


@dataclass
class Prediction:
    score: np.ndarray
    score_std: np.ndarray
    state_proba: np.ndarray | None = None  # cells x classes
    state_classes: np.ndarray | None = None
    # Prediction before clipping to [0, 1], where the model clips. Used for the
    # ensemble spread so that cells clipped at 0 or 1 do not report zero uncertainty.
    raw_score: np.ndarray | None = None


class Model(Protocol):
    name: str

    def fit(self, x: Inputs, y: Targets, seed: int) -> Model: ...

    def predict(self, x: Inputs) -> Prediction: ...

    def pathway_importance(self) -> np.ndarray:
        """Non-negative importance per gene set (global)."""
        ...


class Ensemble:
    """Bootstrap-aggregated ensemble. ``score_std`` = spread across members."""

    def __init__(self, factory: Callable[[], Model], size: int, name: str) -> None:
        self.factory, self.size, self.name = factory, size, name
        self.members: list[Model] = []

    def __getstate__(self) -> dict:
        # The factory is only needed for fitting; dropping it keeps bundles picklable.
        return {k: v for k, v in self.__dict__.items() if k != "factory"}

    def fit(self, x: Inputs, y: Targets, seed: int) -> Ensemble:
        rng = np.random.default_rng(seed)
        self.members = []
        for i in range(self.size):
            idx = rng.integers(0, len(x), len(x)) if self.size > 1 else np.arange(len(x))
            self.members.append(self.factory().fit(x.take(idx), y.take(idx), seed=seed + i))
        return self

    def predict(self, x: Inputs) -> Prediction:
        preds = [m.predict(x) for m in self.members]
        scores = np.stack([p.score for p in preds])
        raw = np.stack([p.score if p.raw_score is None else p.raw_score for p in preds])
        proba = classes = None
        if preds[0].state_proba is not None:
            classes = np.unique(np.concatenate([p.state_classes for p in preds]))
            proba = np.mean([_align_proba(p, classes) for p in preds], axis=0)
        # Ensemble spread (of the unclipped predictions) + mean of members' own
        # (aleatoric) std where provided. The score itself stays the mean of clipped
        # member scores, so reported metrics are unchanged.
        member_std = np.mean([p.score_std for p in preds], axis=0)
        return Prediction(
            score=scores.mean(0),
            score_std=np.sqrt(raw.var(0) + member_std**2),
            state_proba=proba,
            state_classes=classes,
        )

    def pathway_importance(self) -> np.ndarray:
        imp = np.mean([m.pathway_importance() for m in self.members], axis=0)
        return imp / (imp.sum() + 1e-12)


def _align_proba(p: Prediction, classes: np.ndarray | None) -> np.ndarray:
    assert p.state_proba is not None
    assert p.state_classes is not None
    assert classes is not None
    out = np.zeros((len(p.score), len(classes)))
    for j, c in enumerate(p.state_classes):
        out[:, np.where(classes == c)[0][0]] = p.state_proba[:, j]
    return out
