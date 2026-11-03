# parpi-resistance

[![CI](https://github.com/OWNER/parpi-resistance/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/parpi-resistance/actions/workflows/ci.yml)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97-model-yellow)](https://huggingface.co/OWNER/parpi-resistance)
[![Space](https://img.shields.io/badge/%F0%9F%A4%97-demo-blue)](https://huggingface.co/spaces/OWNER/parpi-resistance-app)

Reproducible single-cell models of **PARP-inhibitor (olaparib) resistance** in ovarian
cancer, with an exploratory breast-cancer check.

Give it single-cell RNA-seq counts. For every cell it returns:

- a **resistance score** (0 = drug-naive, 1 = adapted to the top training dose),
- an **uncertainty** for that score,
- the **pathways driving** the score,
- **dose-response curves** of pathway activity across olaparib concentrations, with
  bootstrap bands.

> Research use only. Not a diagnostic or a clinical decision tool.

## Quick start

```bash
# 1. Environment (Python ≥3.10, https://docs.astral.sh/uv/)
make lock            # first time only: creates uv.lock -> commit it
make setup

# 2. Prove the pipeline end to end on synthetic data with planted ground truth (~1 min)
make smoke

# 3. Full run on the public GEO data (downloads ~GBs, see docs/DATA.md first)
make reproduce
make gate            # exits non-zero unless the model passes the release criteria
```

## Use a trained model

```bash
# CLI
parpi predict sample.h5ad -m hf://OWNER/parpi-resistance@v0.1.0 -o predictions.csv
parpi dose-curve -m hf://OWNER/parpi-resistance@v0.1.0 -o curve.csv

# Docker (web app on http://localhost:7860)
docker run --rm -p 7860:7860 -e PARPI_MODEL=hf://OWNER/parpi-resistance@v0.1.0 \
  ghcr.io/OWNER/parpi-resistance:latest
```

```python
from parpi.bundle import ModelBundle
from parpi.data.readers import read_any

bundle = ModelBundle.load("hf://OWNER/parpi-resistance@v0.1.0")
pred = bundle.predict(read_any("sample.h5ad"))  # one row per cell
bundle.pathway_importance().head(10)
```

Accepted inputs: `.h5ad`, 10x `.h5`, 10x matrix folders, and CSV/TSV count tables. Genes
must be HGNC symbols. Raw counts are expected, and QC is applied unless `--no-qc` is set.

## What makes the results trustworthy

| Practice | How it is enforced |
|---|---|
| Every result traces to its inputs | Config hash, git commit, package versions, and data/gene-set SHA-256s are stored with each model |
| Same input, same output | Seeded, deterministic training. A test retrains and asserts identical metrics |
| Honest evaluation | Whole doses are held out (`leave_dose_out`). External cell lines and PDX samples are never used for training or model selection |
| Beats trivial baselines | Every metric is reported next to "predict the mean", "copy the nearest dose" and a sequencing-depth-only model |
| No label leakage | Labels are targets only, never inputs |
| Model choice is earned | Ridge, gradient boosting and the hierarchical network compete. The release model is picked automatically by CV |
| Gated releases | `parpi gate` blocks publishing if the model does not beat baselines or generalise to external ovarian data |
| Numbers are never hand-typed | The model card is generated from the metrics files of the run that produced the model |

Details: [docs/DESIGN.md](docs/DESIGN.md) · data provenance: [docs/DATA.md](docs/DATA.md) ·
release setup: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Models

| Name | Input | Notes |
|---|---|---|
| `ridge` | pathway scores | Linear, exact coefficients, strong baseline |
| `hgb` | pathway scores | Gradient-boosted trees, non-linear baseline |
| `hierarchical` | per-gene ranks | Gene → pathway (learned within-pathway gene weights) → cell (pathway attention). PyTorch |
| dose-response | log-dose | PCHIP interpolation of pathway activity, bootstrap bands, extrapolation flagged |

All models are bootstrap ensembles, which is where the per-cell uncertainty comes from.

## Repository layout

```
configs/            run configs (default = real data, smoke = synthetic) + dataset specs
src/parpi/          package: data/, models/, preprocess, evaluation, pipeline, bundle, cli
app/                Gradio web app (logic.py is unit-tested; app.py is the UI)
tests/              pytest suite (synthetic data with planted signal)
scripts/            Hugging Face Space deployment
.github/workflows/  ci.yml · train.yml (manual real-data run) · release.yml (tag → publish)
Dockerfile          one image for the CLI, batch inference and the web app
dvc.yaml            optional stage caching
```

## Citation

If you use this, cite the data source: França et al., *Cellular adaptation to cancer therapy
along a resistance continuum*, Nature 631, 876–883 (2024). See `CITATION.cff`.
