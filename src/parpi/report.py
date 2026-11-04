"""Generate the Hugging Face model card from the metrics of the run that produced the model.

Numbers in the card are never typed by hand: they are read from metrics files. The
scope ("validated for" / "not validated for") is derived from which datasets gate the
release and what they scored, so the card cannot claim more than the run tested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parpi.config import Config
from parpi.pipeline import PRIMARY_METRIC, paths

PAPER = (
    "Satyal, Long, Gao. *Attention-Based Hierarchical Graph Autoencoder for "
    "Dose-Specific Single-Cell Resistance Dynamics*"
)
DATA_PAPER = (
    "França et al., *Cellular adaptation to cancer therapy along a resistance "
    "continuum*, Nature 631, 876–883 (2024)"
)


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join(
        "| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n" for r in rows
    )


def _line_auroc(r: dict[str, Any]) -> float | None:
    return next((v for k, v in r.items() if k.startswith("auroc_control_vs")), None)


def model_card(cfg: Config, hf_repo: str = "OWNER/parpi-resistance") -> str:
    p = paths(cfg)
    cv = json.loads((p["metrics"] / "cv.json").read_text())
    ext_path = p["metrics"] / "external.json"
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    meta = json.loads((p["models"] / cv["selected_model"] / "metadata.json").read_text())
    best = cv["selected_model"]
    m = cv["models"][best]
    by_line = "per_dataset" in m
    train_types = sorted({d.cancer_type for d in cfg.train_datasets})
    train_names = ", ".join(d.name for d in cfg.train_datasets)

    # ---------------------------------------------------------------- scope
    validated, not_validated = [], []
    if by_line:
        per = m["per_dataset"]
        sps = [r["spearman_dose_means"] for r in per.values()]
        aucs = [a for a in (_line_auroc(r) for r in per.values()) if a is not None]
        validated.append(
            f"Ranking olaparib doses in a {'/'.join(train_types)} cancer cell line it was not "
            f"trained on (leave-one-line-out over {train_names}): dose Spearman "
            f"{min(sps):.2f}–{max(sps):.2f}, AUROC untreated vs top dose "
            f"{min(aucs):.2f}–{max(aucs):.2f}."
        )
    else:
        validated.append(
            f"Ranking held-out olaparib doses within the training system ({train_names}): "
            f"dose Spearman {m[PRIMARY_METRIC]:.2f}."
        )
    for d in cfg.external_datasets:
        res = ext.get(d.name, {}).get(best, {})
        auc = res.get("auroc")
        if auc is None:
            continue
        line = f"`{d.name}` ({d.cancer_type}, {d.accession or d.source}): AUROC {auc:.3f}"
        if cfg.is_gated(d):
            validated.append(line + " (gated).")
        else:
            not_validated.append(line + " (reported, not part of the claim).")
    breast = [d.name for d in cfg.external_datasets if d.cancer_type == "breast"]
    if not breast:
        not_validated.append("Breast cancer: not evaluated.")
    not_validated += [
        "Tumors or patients: the score is not a tumor- or patient-level resistance measure.",
        "Absolute levels: scores shift between cell lines; compare samples within one experiment.",
        "Uncertainty: the ensemble spread is not yet calibrated and understates the real error.",
    ]

    # ------------------------------------------------------------ evaluation
    if by_line:
        line_rows = [
            {
                "held-out line": name,
                "dose Spearman": r["spearman_dose_means"],
                "AUROC untreated vs top": _line_auroc(r),
                "cell MAE": r["mae_cell"],
                "mean baseline MAE": r["mae_predict_train_mean"],
            }
            for name, r in m["per_dataset"].items()
        ]
        cv_block = (
            "Each training cell line is held out in turn; the model is trained on the others.\n\n"
            + _table(
                line_rows,
                [
                    "held-out line",
                    "dose Spearman",
                    "AUROC untreated vs top",
                    "cell MAE",
                    "mean baseline MAE",
                ],
            )
        )
    else:
        cv_rows = [{"model": k, **v} for k, v in cv["models"].items()]
        cv_block = "Whole doses are held out in each fold.\n\n" + _table(
            cv_rows, ["model", "spearman_dose_means", "mae_cell", "mae_predict_train_mean"]
        )
    ext_rows = [
        {
            "dataset": ds,
            "cancer": res["cancer_type"],
            "AUROC": res.get(best, {}).get("auroc") or "n/a (no resistant labels)",
            "cells": res.get(best, {}).get("n_cells") or "—",
            "role": "gated" if res.get("gated", True) else "report only",
        }
        for ds, res in ext.items()
    ]
    bullets = lambda xs: "\n".join(f"- {x}" for x in xs)  # noqa: E731

    return f"""---
license: mit
library_name: parpi-resistance
tags: [single-cell, scRNA-seq, drug-adaptation, PARP-inhibitor, olaparib, ovarian-cancer]
---

# PARP-inhibitor adaptation score for single cells (in vitro)

Scores each cell in a single-cell RNA-seq sample by how far it has adapted to the PARP
inhibitor olaparib (0 = untreated, 1 = adapted to the highest training dose), with the
pathways behind the score. It extends the hierarchical model of {PAPER} to cell-level
learning with held-out-line validation.

**Research use only. Not a diagnostic, and not a tumor or patient resistance score.**

## Scope

**Validated for**

{bullets(validated)}

**Not validated for**

{bullets(not_validated)}

## Released model

- Model: **{best}**, trained on {meta["n_train_cells"]:,} cells from {train_names}
- Doses (µM): {meta["trained_doses_um"]}
- Config fingerprint `{meta["config_fingerprint"]}` · git commit `{meta["git_commit"]}`

## Evaluation

### Cross-validation

{cv_block}

### External datasets

AUROC separates treated/resistant from untreated cells; 0.5 = chance. Only gated
datasets are part of the claim above.

{
        _table(ext_rows, ["dataset", "cancer", "AUROC", "cells", "role"])
        if ext_rows
        else "_No external datasets evaluated._"
    }

## Usage

```python
from parpi.bundle import ModelBundle
from parpi.data.readers import read_any

bundle = ModelBundle.load("hf://{hf_repo}")
predictions = bundle.predict(read_any("my_sample.h5ad"))  # one row per cell
```

Input: raw counts, HGNC gene symbols, human cells; at least
{cfg.features.min_gene_coverage:.0%} of the model's genes must be present. Compare
samples within one experiment (for example treated vs untreated of the same line).

## Data and citation

Training and test data: {DATA_PAPER} (GEO SuperSeries GSE247691). Model architecture:
{PAPER}.

## Reproduce

```bash
git clone https://github.com/sachitsatyal/parpi-resistance && cd parpi-resistance
uv sync --all-extras
uv run parpi reproduce -c configs/release.yaml
```
"""


def write_model_card(cfg: Config, hf_repo: str = "OWNER/parpi-resistance") -> Path:
    p = paths(cfg)
    card = model_card(cfg, hf_repo)
    sel = (p["models"] / "SELECTED").read_text().strip()
    out = p["models"] / sel / "README.md"
    out.write_text(card, encoding="utf-8")
    return out
