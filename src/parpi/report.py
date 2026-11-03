"""Generate the Hugging Face model card from the metrics of the run that produced the model.

Numbers in the card are never typed by hand: they are read from metrics files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parpi.config import Config
from parpi.pipeline import PRIMARY_METRIC, paths


def _fmt(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join(
        "| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |\n" for r in rows
    )


def model_card(cfg: Config, hf_repo: str = "OWNER/parpi-resistance") -> str:
    p = paths(cfg)
    cv = json.loads((p["metrics"] / "cv.json").read_text())
    ext_path = p["metrics"] / "external.json"
    ext = json.loads(ext_path.read_text()) if ext_path.exists() else {}
    summary = json.loads((p["processed"] / "summary.json").read_text())
    meta = json.loads((p["models"] / cv["selected_model"] / "metadata.json").read_text())

    cv_rows = [{"model": k, **v} for k, v in cv["models"].items()]
    cv_cols = [
        "model",
        "spearman_dose_means",
        "spearman_cell",
        "mae_cell",
        "mae_predict_train_mean",
    ]
    if any("state_balanced_accuracy" in r for r in cv_rows):
        cv_cols += ["state_balanced_accuracy", "state_chance_balanced_accuracy", "state_ece"]

    ext_rows = []
    for ds, res in ext.items():
        for model, m in res.items():
            if isinstance(m, dict):
                ext_rows.append(
                    {
                        "dataset": ds,
                        "cancer": res["cancer_type"],
                        "model": model,
                        "auroc": m.get("auroc", "n/a"),
                        "n_cells": m.get("n_cells"),
                        "gated": "yes" if res.get("gated", True) else "report only",
                    }
                )
    dr = cv.get("dose_response_loo", {})
    datasets = "\n".join(
        f"- `{d.name}` ({d.role}, {d.cancer_type}): {d.accession or d.source}" for d in cfg.datasets
    )
    breast = [d.name for d in cfg.external_datasets if d.cancer_type == "breast"]

    return f"""---
license: mit
library_name: parpi-resistance
tags: [single-cell, scRNA-seq, drug-resistance, PARP-inhibitor, olaparib, ovarian-cancer]
---

# PARP-inhibitor resistance score (single-cell)

Predicts, for each cell in a single-cell RNA-seq sample, how far along the
olaparib resistance continuum it sits (0 = drug-naive, 1 = adapted to the
highest training dose), with an uncertainty estimate and the pathways driving
the prediction. A companion dose-response model gives pathway activity as a
smooth function of olaparib concentration within the measured range.

**Research use only. Not a diagnostic.**

## Released model

- Selected model: **{cv["selected_model"]}** (chosen automatically by
  `{PRIMARY_METRIC}` under leave-dose-out cross-validation)
- Config fingerprint: `{meta["config_fingerprint"]}` · git commit: `{meta["git_commit"]}`
- Trained on {meta["n_train_cells"]} cells; doses (µM): {meta["trained_doses_um"]}

## Training data

{datasets}

Preprocessing summary: `{json.dumps(summary.get("train", {}), default=str)}`

## Evaluation

### Held-out doses (cross-validation)

Whole doses are held out in each fold (held-out doses per fold:
{cv["folds_held_out_doses"]}). `mae_predict_train_mean` is the error of
always predicting the average — any useful model must beat it. The
`depth_only` row uses sequencing depth alone and should fail; if it does not,
labels are confounded with technical factors.

{_table(cv_rows, cv_cols)}

### Unseen datasets (external validation)

AUROC for separating resistant from parental cells. 0.5 = chance.

{
        _table(ext_rows, ["dataset", "cancer", "model", "auroc", "n_cells", "gated"])
        if ext_rows
        else "_No external datasets evaluated._"
    }

### Dose-response interpolation (leave-one-dose-out)

Mean absolute error of the predicted pathway profile at a held-out dose:
model {_fmt(dr.get("model_mae", float("nan")))},
copy-nearest-dose baseline {_fmt(dr.get("nearest_dose_mae", float("nan")))},
linear interpolation {_fmt(dr.get("linear_interp_mae", float("nan")))}.

## Intended use and limits

- Input: raw counts, genes as HGNC symbols, human cells. At least
  {cfg.features.min_gene_coverage:.0%} of the model's genes must be present.
- The score is relative to the training escalation (one drug, olaparib). It is
  not a clinical response probability.
- Dose-response predictions outside the measured dose range are returned with
  `extrapolated=True` and have not been validated.
- Breast cancer: {
        "evaluated only on " + ", ".join(breast) + " (one cell line, a different "
        "PARP inhibitor). Treat breast predictions as exploratory."
        if breast
        else "not evaluated. Do not use on breast cancer data."
    }

## Usage

```python
from parpi.bundle import ModelBundle
from parpi.data.readers import read_any

bundle = ModelBundle.load("hf://{hf_repo}")
cells = read_any("my_sample.h5ad")
predictions = bundle.predict(cells)
```

## Reproduce

```bash
git clone <repo> && cd parpi-resistance
make reproduce CONFIG=configs/default.yaml
```
"""


def write_model_card(cfg: Config, hf_repo: str = "OWNER/parpi-resistance") -> Path:
    p = paths(cfg)
    card = model_card(cfg, hf_repo)
    sel = (p["models"] / "SELECTED").read_text().strip()
    out = p["models"] / sel / "README.md"
    out.write_text(card)
    return out
