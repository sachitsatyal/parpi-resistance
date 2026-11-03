"""UI-independent logic for the web app (unit-tested without Gradio)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parpi.bundle import ModelBundle
from parpi.data.readers import read_any

MAX_CELLS = int(os.environ.get("PARPI_MAX_CELLS", "50000"))


@lru_cache(maxsize=2)
def load_bundle(location: str | None = None) -> ModelBundle:
    """Model location from argument or $PARPI_MODEL (local folder or hf://org/repo@rev)."""
    loc = location or os.environ.get("PARPI_MODEL", "hf://OWNER/parpi-resistance")
    return ModelBundle.load(loc)


def score_file(
    path: str | Path, bundle: ModelBundle, apply_qc: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (per-cell predictions, per-sample summary, markdown status)."""
    cd = read_any(path)
    if cd.n_cells > MAX_CELLS:
        raise ValueError(
            f"{cd.n_cells} cells exceeds the demo limit of {MAX_CELLS}; "
            "use the CLI or Docker image for large files"
        )
    pred = bundle.predict(cd, apply_qc=apply_qc)
    summary = (
        pred.groupby("sample")["resistance_score"]
        .agg(cells="size", mean="mean", median="median", std="std")
        .round(3)
        .reset_index()
    )
    status = (
        f"Scored **{len(pred)}** cells ({cd.n_cells - len(pred)} removed by QC). "
        f"Gene coverage **{pred.attrs['gene_coverage']:.0%}**. "
        f"Model: `{bundle.metadata.get('model')}` · config "
        f"`{bundle.metadata.get('config_fingerprint')}`."
    )
    if pred.attrs["gene_coverage"] < 0.8:
        status += "\n\n⚠️ Low gene coverage: check that genes are HGNC symbols."
    return pred, summary, status


def score_histogram(pred: pd.DataFrame):  # type: ignore[no-untyped-def]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for sample, grp in pred.groupby("sample"):
        ax.hist(grp["resistance_score"], bins=np.linspace(0, 1, 31), alpha=0.6, label=str(sample))
    ax.set_xlabel("Resistance score (0 = drug-naive, 1 = top training dose)")
    ax.set_ylabel("Cells")
    if pred["sample"].nunique() <= 10:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def dose_curve_figure(bundle: ModelBundle, pathways: list[str], max_dose: float = 640.0):  # type: ignore[no-untyped-def]
    trained = np.array(bundle.metadata.get("trained_doses_um", [0, 320]), dtype=float)
    doses = np.unique(np.concatenate([np.geomspace(0.5, max_dose, 80), trained, [0.0]]))
    curve = bundle.dose_curve(doses, pathways)
    lo, hi = bundle.dose_response.predict_interval(doses)  # type: ignore[union-attr]
    names = list(bundle.features.gene_sets.names)
    x = np.log2(doses + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for p in pathways:
        k = names.index(p)
        (line,) = ax.plot(x, curve[p], label=p[:45])
        ax.fill_between(x, lo[:, k], hi[:, k], color=line.get_color(), alpha=0.2, linewidth=0)
    extrap = curve["extrapolated"].to_numpy()
    if extrap.any():
        ax.axvspan(x[~extrap].max(), x.max(), color="grey", alpha=0.15, label="not measured")
    ticks = [0, 1, 5, 20, 80, 320]
    ax.set_xticks(np.log2(np.array(ticks) + 1), [str(t) for t in ticks])
    ax.set_xlabel("Olaparib (µM, log scale)")
    ax.set_ylabel("Pathway activity (rank score)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
