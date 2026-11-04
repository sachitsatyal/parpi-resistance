"""End-to-end pipeline stages: prepare -> train (CV + final fit) -> evaluate -> report.

Each stage reads only files written by earlier stages, so stages can be cached
and re-run independently (see ``dvc.yaml``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from parpi.config import Config
from parpi.data.datasets import load_dataset, resistance_target
from parpi.data.synthetic import make_gene_sets
from parpi.evaluation import (
    classification_metrics,
    external_metrics,
    make_folds,
    regression_metrics,
)
from parpi.genesets import GeneSets, load_gene_sets
from parpi.models.base import Ensemble, Inputs, Model, Targets
from parpi.models.baselines import DepthOnlyModel, HGBModel, RidgeModel
from parpi.models.dose_response import DoseResponseModel, leave_one_dose_out
from parpi.preprocess import FeatureSpace, filter_cells
from parpi.seed import seed_everything

log = logging.getLogger(__name__)

PRIMARY_METRIC = "spearman_dose_means"


# ---------------------------------------------------------------- utilities
def paths(cfg: Config) -> dict[str, Path]:
    data, out = Path(cfg.data_dir), Path(cfg.output_dir)
    return {
        "data": data,
        "processed": Path(cfg.processed_dir) if cfg.processed_dir else data / "processed",
        "genesets": data / "genesets",
        "out": out,
        "models": out / "models",
        "metrics": out / "metrics",
    }


def _library(cfg: Config) -> GeneSets:
    if cfg.features.gene_sets == "synthetic":
        return make_gene_sets()[1]
    return load_gene_sets(cfg.features.gene_sets, paths(cfg)["genesets"])


def _save_processed(folder: Path, x: Inputs, obs: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "ranks.npy", x.ranks.astype(np.float16))
    np.save(folder / "pathways.npy", x.pathways.astype(np.float32))
    obs.to_csv(folder / "obs.csv", index=False)


def load_processed(folder: Path) -> tuple[Inputs, pd.DataFrame]:
    x = Inputs(np.load(folder / "ranks.npy").astype(np.float32), np.load(folder / "pathways.npy"))
    return x, pd.read_csv(folder / "obs.csv")


def model_factory(name: str, cfg: Config, fs: FeatureSpace) -> Callable[[], Model]:
    if name == "ridge":
        return lambda: RidgeModel(cfg.train.baseline)
    if name == "hgb":
        return lambda: HGBModel(cfg.train.baseline)
    if name == "hierarchical":
        from parpi.models.hierarchical import HierarchicalModel  # needs torch

        membership = fs.membership
        return lambda: HierarchicalModel(cfg.train.hierarchical, membership)
    raise ValueError(f"unknown model {name}")


def _available_models(cfg: Config) -> list[str]:
    names: list[str] = list(cfg.train.models)
    if "hierarchical" in names:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("PyTorch not installed: skipping the hierarchical model")
            names.remove("hierarchical")
    return names


def _targets(obs: pd.DataFrame, use_states: bool = True) -> Targets:
    dose = obs["dose_um"].to_numpy(float)
    state = None
    if use_states and "state" in obs and obs["state"].notna().any():
        # Missing states (e.g. control cells) become "": excluded from the state loss.
        state = np.where(obs["state"].isna(), "", obs["state"].astype(str)).astype(str)
    return Targets(resistance_target(dose, dose.max()), state)


# ------------------------------------------------------------------ prepare
def prepare(cfg: Config) -> None:
    p = paths(cfg)
    seed_everything(cfg.train.seed)
    library = _library(cfg)

    train_parts = []
    for d in cfg.train_datasets:
        cd = filter_cells(load_dataset(d, p["data"]), cfg.qc)
        if "dose_um" not in cd.obs:
            raise ValueError(f"training dataset {d.name} needs a dose_map")
        train_parts.append(cd)
    from parpi.celldata import CellData

    universe = np.array(sorted(set().union(*[set(c.genes) for c in train_parts])))
    train = CellData.concat([c.align_genes(universe)[0] for c in train_parts])
    fs = FeatureSpace.fit(train, library, cfg.qc, cfg.features)
    fs.save(p["processed"] / "feature_space")

    _, ranks, cov = fs.transform(train, apply_qc=False)
    _save_processed(p["processed"] / "train", Inputs(ranks, fs.pathway_scores(ranks)), train.obs)
    qc_cols = [c for c in ("total_counts", "n_genes", "pct_mito") if c in train.obs]
    # With several training datasets, condition names repeat ("C", "T5" ...):
    # label each group "dataset/condition" when more than one dataset is present.
    multi = train.obs["dataset"].nunique() > 1
    group = (
        train.obs["dataset"].astype(str) + "/" + train.obs["condition"].astype(str)
        if multi
        else train.obs["condition"].astype(str)
    )
    by_cond = train.obs.groupby(group)
    qc_by_condition = by_cond[qc_cols].median().round(2)
    qc_by_condition["cells"] = by_cond.size()
    if "state" in train.obs:
        qc_by_condition["cells_with_state"] = by_cond["state"].count()
    summary: dict[str, Any] = {
        "train": {
            "cells": train.n_cells,
            "gene_coverage": cov,
            "conditions": group.value_counts().to_dict(),
            # Per-dose QC medians: a library that differs from its neighbours
            # (e.g. a re-sequenced dose) shows up here before it biases a model.
            "qc_by_condition": qc_by_condition.to_dict(orient="index"),
        }
    }
    for d in cfg.external_datasets:
        cd = load_dataset(d, p["data"])
        qc_cd, ranks, cov = fs.transform(cd)
        _save_processed(p["processed"] / d.name, Inputs(ranks, fs.pathway_scores(ranks)), qc_cd.obs)
        summary[d.name] = {"cells": qc_cd.n_cells, "gene_coverage": cov}
    (p["processed"] / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("prepared: %s", summary)


# -------------------------------------------------------------------- train
def cross_validate(cfg: Config, x: Inputs, obs: pd.DataFrame, fs: FeatureSpace) -> dict[str, Any]:
    y = _targets(obs, cfg.train.predict_states)
    dose = obs["dose_um"].to_numpy(float)
    by_dataset = cfg.split.strategy == "leave_dataset_out"
    groups = obs["dataset"].astype(str).to_numpy() if by_dataset else dose
    folds = make_folds(groups, cfg.split, cfg.train.seed)
    results: dict[str, Any] = {}
    oof_frames = []
    mean_base = np.zeros(len(x))
    for test in folds:
        mean_base[test] = np.delete(y.score, test).mean()
    for name in _available_models(cfg):
        oof = np.zeros(len(x))
        oof_std = np.zeros(len(x))
        proba_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        t0 = time.time()
        for k, test in enumerate(folds):
            train_idx = np.setdiff1d(np.arange(len(x)), test)
            ens = Ensemble(model_factory(name, cfg, fs), cfg.train.ensemble_size, name)
            ens.fit(x.take(train_idx), y.take(train_idx), seed=cfg.train.seed + 100 * k)
            pred = ens.predict(x.take(test))
            oof[test], oof_std[test] = pred.score, pred.score_std
            held = (
                sorted(np.unique(groups[test]).tolist())
                if by_dataset
                else sorted(np.unique(dose[test]).tolist())
            )
            log.info(
                "CV %s: fold %d/%d done (held out %s, %.0fs elapsed)",
                name,
                k + 1,
                len(folds),
                held,
                time.time() - t0,
            )
            if (
                pred.state_proba is not None
                and pred.state_classes is not None
                and y.state is not None
            ):
                proba_parts.append((pred.state_proba, pred.state_classes, y.state[test]))
        metrics = regression_metrics(oof, y.score, dose, mean_base)
        if proba_parts:
            # Score each fold separately: class sets can differ between folds.
            # Folds without labelled cells (e.g. a held-out line with no states) are skipped.
            scored = [classification_metrics(pp, cc, tt) for pp, cc, tt in proba_parts]
            per_fold = [m for m in scored if m]
            if per_fold:
                metrics.update(
                    {k: float(np.nanmean([m[k] for m in per_fold])) for k in per_fold[0]}
                )
        if by_dataset:
            metrics.update(_per_dataset_metrics(oof, y.score, dose, groups, mean_base))
        metrics["seconds"] = round(time.time() - t0, 1)
        results[name] = metrics
        oof_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "dose_um": dose,
                    "target": y.score,
                    "pred": oof,
                    "pred_std": oof_std,
                }
            )
        )
        log.info("CV %s: %s", name, metrics)

    # Sanity baseline: sequencing depth alone.
    if "total_counts" in obs:
        depth = obs["total_counts"].to_numpy(float)
        oof = np.zeros(len(x))
        for test in folds:
            tr = np.setdiff1d(np.arange(len(x)), test)
            oof[test] = DepthOnlyModel().fit(depth[tr], y.score[tr]).predict(depth[test])
        dm = regression_metrics(oof, y.score, dose, mean_base)
        # Pooled correlations of a near-constant predictor only reflect per-fold
        # intercepts under leave-dose-out CV; MAE vs. the mean baseline is the test.
        results["depth_only"] = {k: dm[k] for k in ("mae_cell", "mae_predict_train_mean")}
    results["_oof"] = pd.concat(oof_frames, ignore_index=True)
    results["_folds"] = [sorted(np.unique(dose[f]).tolist()) for f in folds]
    return results


def _per_dataset_metrics(
    pred: np.ndarray, target: np.ndarray, dose: np.ndarray, groups: np.ndarray, base: np.ndarray
) -> dict[str, Any]:
    """Metrics for each held-out dataset, plus their mean (the leave-dataset-out score)."""
    per: dict[str, dict[str, float]] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        r = regression_metrics(pred[m], target[m], dose[m], base[m])
        top = dose[m].max()
        sel = m & ((dose == 0) | (dose == top))
        r.update(
            {
                f"auroc_control_vs_{top:g}uM": external_metrics(
                    pred[sel], (dose[sel] == top).astype(float)
                ).get("auroc", float("nan"))
            }
        )
        per[g] = r
    return {
        "per_dataset": per,
        "spearman_dose_means_by_dataset": float(
            np.nanmean([r["spearman_dose_means"] for r in per.values()])
        ),
        "mae_cell_by_dataset": float(np.nanmean([r["mae_cell"] for r in per.values()])),
    }


def select_model(cv: dict[str, Any], candidates: list[str]) -> str:
    """Highest dose-level Spearman; ties broken by lower cell MAE, then simpler model.

    Under leave-dataset-out CV the per-held-out-dataset averages are used.
    """
    order = {"ridge": 0, "hgb": 1, "hierarchical": 2}

    def key(m: str) -> tuple[float, float, int]:
        c = cv[m]
        sp = c.get("spearman_dose_means_by_dataset", c[PRIMARY_METRIC])
        mae = c.get("mae_cell_by_dataset", c["mae_cell"])
        return (-round(sp, 3), round(mae, 3), order[m])

    return sorted(candidates, key=key)[0]


def train(cfg: Config) -> None:
    from parpi.bundle import ModelBundle

    p = paths(cfg)
    seed_everything(cfg.train.seed)
    fs = FeatureSpace.load(p["processed"] / "feature_space")
    x, obs = load_processed(p["processed"] / "train")
    cv = cross_validate(cfg, x, obs, fs)
    p["metrics"].mkdir(parents=True, exist_ok=True)
    cv.pop("_oof").to_csv(p["metrics"] / "cv_predictions.csv", index=False)
    folds = cv.pop("_folds")
    models = _available_models(cfg)
    best = select_model(cv, models)

    dr = DoseResponseModel().fit(obs["dose_um"].to_numpy(float), x.pathways)
    dr_bench = leave_one_dose_out(obs["dose_um"].to_numpy(float), x.pathways)
    dr_bench.to_csv(p["metrics"] / "dose_response_loo.csv", index=False)

    y = _targets(obs, cfg.train.predict_states)
    for name in models:
        log.info("final fit: %s on all %d cells", name, len(x))
        ens = Ensemble(model_factory(name, cfg, fs), cfg.train.ensemble_size, name)
        ens.fit(x, y, seed=cfg.train.seed)
        ens.pathway_importance()  # compute once now so it is stored inside the bundle
        bundle = ModelBundle(
            fs,
            ens,
            dr,
            metadata={
                "model": name,
                "selected": name == best,
                "config_fingerprint": cfg.fingerprint(),
                "config": cfg.model_dump(mode="json"),
                "cv_metrics": cv[name],
                "max_dose_um": float(obs["dose_um"].max()),
                "trained_doses_um": sorted(obs["dose_um"].unique().tolist()),
                "pathway_mean": x.pathways.mean(0).tolist(),
                "pathway_std": x.pathways.std(0).tolist(),
                "n_train_cells": len(x),
            },
        )
        bundle.save(p["models"] / name)
        bundle.pathway_importance().to_csv(
            p["metrics"] / f"pathway_importance_{name}.csv", index=False
        )
    (p["metrics"] / "cv.json").write_text(
        json.dumps(
            {
                "selected_model": best,
                "primary_metric": PRIMARY_METRIC,
                "folds_held_out_doses": folds,
                "models": cv,
                "dose_response_loo": dr_bench.mean(numeric_only=True).to_dict(),
            },
            indent=2,
        )
    )
    (p["models"] / "SELECTED").write_text(best + "\n")
    log.info("selected model: %s", best)


# ----------------------------------------------------------------- evaluate
def evaluate_external(cfg: Config) -> dict[str, Any]:
    from parpi.bundle import ModelBundle

    p = paths(cfg)
    results: dict[str, Any] = {}
    for d in cfg.external_datasets:
        x, obs = load_processed(p["processed"] / d.name)
        results[d.name] = {"cancer_type": d.cancer_type, "gated": cfg.is_gated(d)}
        for name in _available_models(cfg):
            bundle = ModelBundle.load(p["models"] / name)
            pred = bundle.model.predict(x)
            m = external_metrics(
                pred.score, obs.get("resistant", pd.Series(np.nan, index=obs.index)).to_numpy()
            )
            m["mean_score_by_condition"] = (
                pd.Series(pred.score).groupby(obs["condition"]).mean().round(3).to_dict()
            )
            results[d.name][name] = m
    p["metrics"].mkdir(parents=True, exist_ok=True)
    (p["metrics"] / "external.json").write_text(json.dumps(results, indent=2))
    return results


# --------------------------------------------------------------------- gate
def quality_gate(
    cfg: Config,
    min_spearman: float = 0.8,
    max_mae_ratio: float = 0.8,
    min_external_auroc: float = 0.65,
    min_line_auroc: float = 0.7,
) -> list[str]:
    """Return a list of failures; an empty list means the release may proceed.

    Leave-dose-out configs (one training system):
      * dose-level Spearman of the selected model >= ``min_spearman``,
      * cell-level MAE <= ``max_mae_ratio`` x the out-of-fold mean baseline.
    Leave-dataset-out configs (several cell lines; the claim is a *ranking* that
    transfers to a new line, not a calibrated absolute value):
      * mean per-held-out-line dose Spearman >= ``min_spearman``,
      * AUROC control vs top dose >= ``min_line_auroc`` in *every* held-out line.
    Both: every gated external dataset (``Config.is_gated``) of the training cancer
    type must reach AUROC >= ``min_external_auroc``. Others are reported only.
    """
    p = paths(cfg)
    cv = json.loads((p["metrics"] / "cv.json").read_text())
    best = cv["selected_model"]
    m = cv["models"][best]
    failures = []
    if "per_dataset" in m:
        sp = m["spearman_dose_means_by_dataset"]
        if not sp >= min_spearman:
            failures.append(f"{best}: mean per-line dose Spearman={sp:.3f} < {min_spearman}")
        for line, r in m["per_dataset"].items():
            auc = next((v for k, v in r.items() if k.startswith("auroc_control_vs")), None)
            if auc is None or not auc >= min_line_auroc:
                failures.append(
                    f"{best}, held-out {line}: AUROC control vs top={auc} < {min_line_auroc}"
                )
    else:
        if not m[PRIMARY_METRIC] >= min_spearman:
            failures.append(f"{best}: {PRIMARY_METRIC}={m[PRIMARY_METRIC]:.3f} < {min_spearman}")
        if not m["mae_cell"] <= max_mae_ratio * m["mae_predict_train_mean"]:
            failures.append(
                f"{best}: mae_cell={m['mae_cell']:.3f} is not < {max_mae_ratio} x mean baseline "
                f"({m['mae_predict_train_mean']:.3f})"
            )
    ext_path = p["metrics"] / "external.json"
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    train_types = {d.cancer_type for d in cfg.train_datasets}
    for d in cfg.external_datasets:
        res = ext.get(d.name, {}).get(best, {})
        if cfg.is_gated(d) and d.cancer_type in train_types:
            auc = res.get("auroc")
            if auc is None or not auc >= min_external_auroc:
                failures.append(f"{best} on {d.name}: auroc={auc} < {min_external_auroc}")
    return failures
