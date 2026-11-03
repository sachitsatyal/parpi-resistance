"""Splits and metrics, including the trivial baselines every claim is compared against."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)

from parpi.config import SplitConfig


def make_folds(groups: np.ndarray, cfg: SplitConfig, seed: int) -> list[np.ndarray]:
    """Return a list of held-out index arrays.

    ``leave_dose_out``: whole conditions are held out. Conditions are sorted by
    dose and dealt round-robin into folds, so every fold mixes low and high doses
    and covers both interpolation and (at the ends) extrapolation.
    """
    if cfg.strategy == "leave_dataset_out":
        levels = sorted(set(groups.tolist()))
        if len(levels) < 2:
            raise ValueError("leave_dataset_out needs at least two training datasets")
        return [np.where(groups == lvl)[0] for lvl in levels]
    if cfg.strategy == "random_cells":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(groups))
        return [np.sort(f) for f in np.array_split(perm, cfg.n_folds)]
    levels = list(dict.fromkeys(groups[np.argsort(groups, kind="stable")]))
    n_folds = min(cfg.n_folds, len(levels))
    if n_folds < 2:
        raise ValueError("leave_dose_out needs at least two distinct conditions")
    fold_of = {lvl: i % n_folds for i, lvl in enumerate(levels)}
    fold_id = np.array([fold_of[g] for g in groups])
    return [np.where(fold_id == f)[0] for f in range(n_folds)]


def _safe(fn, *args) -> float:  # type: ignore[no-untyped-def]
    try:
        v = float(fn(*args))
    except ValueError:
        return float("nan")
    return v


def regression_metrics(
    pred: np.ndarray, target: np.ndarray, dose: np.ndarray, mean_baseline: np.ndarray | None = None
) -> dict[str, float]:
    """Cell-level and dose-level agreement between predicted score and dose position.

    ``mean_baseline`` should be the out-of-fold "predict the training mean"
    prediction, computed with the same folds as ``pred`` so the comparison is fair.
    """
    if mean_baseline is None:
        mean_baseline = np.full_like(target, target.mean())
    df = pd.DataFrame({"p": pred, "t": target, "d": dose})
    per_dose = df.groupby("d").agg(p=("p", "mean"), t=("t", "first"))
    return {
        "spearman_cell": _safe(lambda: spearmanr(pred, dose).statistic),
        "pearson_cell": _safe(lambda: pearsonr(pred, target).statistic),
        "mae_cell": float(mean_absolute_error(target, pred)),
        "spearman_dose_means": _safe(lambda: spearmanr(per_dose["p"], per_dose.index).statistic),
        "mae_dose_means": float(mean_absolute_error(per_dose["t"], per_dose["p"])),
        "mae_predict_train_mean": float(mean_absolute_error(target, mean_baseline)),
    }


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(1)
    correct = proba.argmax(1) == y_idx
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(
    proba: np.ndarray, classes: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    """State metrics on the labelled cells; {} when the fold has no labelled cells."""
    known = np.isin(truth, classes) & (truth != "")
    if not known.any():
        return {}
    proba, truth = proba[known], truth[known]
    pred = classes[proba.argmax(1)]
    y_idx = np.searchsorted(classes, truth)
    majority = pd.Series(truth).value_counts(normalize=True).iloc[0]
    return {
        "state_balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "state_macro_f1": float(f1_score(truth, pred, average="macro")),
        "state_ece": expected_calibration_error(proba, y_idx),
        "state_majority_class_accuracy": float(majority),
        "state_chance_balanced_accuracy": 1.0 / len(np.unique(truth)),
    }


def external_metrics(score: np.ndarray, resistant: np.ndarray) -> dict[str, float]:
    """Does the score separate resistant from parental cells in an unseen dataset?"""
    ok = ~pd.isna(resistant)
    y = np.asarray(resistant[ok], dtype=int)
    s = score[ok]
    out: dict[str, float] = {"n_cells": int(ok.sum())}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["mean_score_resistant"] = float(s[y == 1].mean())
        out["mean_score_parental"] = float(s[y == 0].mean())
    return out
