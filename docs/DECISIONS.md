# parpi-resistance — decision log

Every choice made while rebuilding the hierarchical single-cell olaparib project, and why.
Newest entries are at the bottom. Last updated: 2026-11-02.

## 1. Assessment of the original project

The starting point was *Attention-Based Hierarchical Graph Autoencoder for Dose-Specific
Single-Cell Resistance Dynamics* (Satyal, Long, Gao).

| Problem found | Why it matters |
|---|---|
| One graph per dose, so the dose-level model learns from **9 samples** | The LSTM, dose embeddings and extrapolation rest on 9 points |
| T640 "extrapolation" appends an untrained embedding slot | There is no data behind the T640 prediction |
| Reported T320 prediction (cos-sim 0.983) is worse than copying T160 (≈0.998, from Fig. 8 drift 0.0023) | The headline result loses to a trivial baseline |
| Resistance labels are model **inputs** (5 of 7 cell features) and also the **target** | Label leakage |
| The autoencoder reconstructs its own embeddings | Low loss is not evidence of biology |
| DEGs and enrichment re-run per dose, so every graph has different nodes | Dose clusters may reflect node sets rather than biology |
| One cell line only; CPA comparison not like-for-like | No evidence the model generalises |

**Decision:** rebuild rather than refactor. Keep the core idea (gene → pathway → cell
hierarchy with attention) and fix the rest.

## 2. Scope choices (asked and answered)

| Question | Choice | Reason |
|---|---|---|
| Starting point | **Fresh repository** | The original's structure carries the problems above |
| What users get | **All three:** per-cell resistance score, dose-response curves, pathway explanations | Requested |
| Breast cancer | **Find data first, then decide** | Avoid claiming breast performance without breast data |
| Graph model | **Baselines + graph model**; the graph model ships only if it wins | Complexity has to earn its place |

## 3. Data decisions

| Decision | Reason |
|---|---|
| Train on **Kuramochi, GSE206125** (França et al., Nature 2024) | The only public scRNA-seq olaparib dose escalation (C, then T1–T320) |
| External ovarian test sets come from the same SuperSeries **GSE247691**: PDX (GSE247690), plus COV362 once its accession is confirmed | Same drug and lab but different models, so a true out-of-sample test |
| Breast check: **GSE202435** (HCC1806 TNBC, parental vs talazoparib-resistant), labelled **exploratory** | No public breast olaparib dose-escalation scRNA-seq exists. This set uses a different PARP inhibitor and only 2 conditions |
| Breast results are reported but do **not** block releases | One cell line with a different drug is too weak to gate on |
| Every download is checksummed (SHA-256 in `MANIFEST.json`) | Silent upstream changes fail loudly instead of changing results |
| Gene sets: **MSigDB Hallmark 2020 + Reactome 2022** via Enrichr, pinned by SHA-256 and saved inside each model | Identical pathway definitions for every dataset; no dependence on later downloads |

## 4. Modelling decisions

| Decision | Reason |
|---|---|
| The unit of learning is the **cell**, not the dose graph | Thousands of samples instead of 9 |
| Target = **log-dose position** `log2(dose+1)/log2(max+1)`; S1–S5 states are an optional extra target | Defined for every cell, no clustering needed, no leakage |
| Features = **within-cell rank pathway scores** (singscore-style) | No dataset-level statistics, so a single sample can be scored. Depth-invariant (tested). Same transform at training and inference |
| Three competing models: **ridge**, **gradient boosting (HGB)**, **hierarchical pathway-attention network** | Strong simple baselines plus the original idea |
| All models are **bootstrap ensembles** | Gives a per-cell uncertainty |
| Dose-response: **PCHIP** interpolation over log-dose, bootstrap bands, out-of-range doses flagged `extrapolated` | A global spline lost to plain linear interpolation in testing; PCHIP beat both trivial predictors (MAE 0.005 vs 0.010 nearest-dose and 0.006 linear) |
| The hierarchical layer works on membership **edges** only (sparse gather), not a dense genes × pathways tensor | The dense version would take about 8 h per CI run at Reactome scale |
| Gene alignment uses a sparse selection-matrix product | The LIL-matrix version was too slow at 8k cells × 20k genes (now 0.15 s) |
| Baseline pathway importance is computed **lazily** with permutation importance on 1,000 cells | Computing it for every fit made HGB cross-validation far too slow |

## 5. Evaluation decisions

| Decision | Reason |
|---|---|
| **Leave-dose-out CV**: doses sorted, dealt round-robin into 5 folds | Tests unseen doses. Every fold mixes low and high doses |
| The "predict the mean" baseline is computed **out-of-fold** | Pooled correlations of a constant predictor are fake under grouped CV (first run showed a spurious −0.49) |
| **Depth-only sanity model** is judged by error, not correlation | Detects technical confounding |
| Model selection: dose-level Spearman, then MAE, then the simpler model | Automatic and reproducible |
| **Release gate**: dose Spearman ≥ 0.8; MAE ≤ 0.8 × mean baseline; external ovarian AUROC ≥ 0.65 | Nothing is published unless it beats trivial baselines and generalises |
| The model card is generated from metrics files | Numbers are never hand-typed |

## 6. Engineering decisions

| Decision | Reason |
|---|---|
| `uv` + `uv.lock`; CPU PyTorch wheels on Linux | Exact, reproducible installs; a small Docker image |
| Typed YAML config (pydantic); config hash stored with every model | Every result traces to its settings |
| Tests on **synthetic data with a planted signal**, including a determinism test | Proves the pipeline recovers known truth and is reproducible |
| GitHub Actions: `ci.yml` (lint, types, tests, smoke run, Docker), `train.yml` (manual real-data run), `release.yml` (tag → retrain → gate → publish) | Industry-standard CI/CD |
| Publishing: model → **Hugging Face Hub** (version-tagged), image → **GHCR**, app → **HF Space** pinned to the same model tag | The app and the model card always describe the same model |
| Gradio app with logic split into `app/logic.py` | The app logic is unit-tested without a browser |

## 7. Run log on Sachit's workstation (Windows, `C:\Users\sachit_satyal\parpi-resistance`)

| Step | Result |
|---|---|
| Installed `uv` 0.13.6 | OK |
| `uv lock` | 202 packages resolved. **Commit `uv.lock`** |
| `uv sync --all-extras` | OK (Python 3.12.14) |
| `uv run pytest` | **34 passed**, including the first-ever run of the graph-model tests |
| Smoke pipeline, first run | Works. **Graph model barely learned** (MAE 0.262 vs mean baseline 0.278) |
| Fix: standardise pathway activations per pathway, add a linear skip path, start the score at the training mean, raise lr to 3e-3 | The raw activations differ by only about 0.01–0.1 and were swamped by layer biases |
| Smoke pipeline after the fix | Graph model MAE **0.130**, dose Spearman **0.90**, external AUROC 1.0. Ridge still selected (0.092), as expected because the synthetic signal is linear by construction |
| `parpi gate` (smoke) | **Passed** |
| Note | A browser download saved the update as `parpi-resistance (1).zip`. Always unzip the newest file |
| `parpi inspect GSE206125` | Four files: one combined count matrix `..._kuramochi_all_C_to_T320.csv.gz`, per-cell metadata `..._metadata_g1.tsv.gz`, and two barcode lists (not needed) |
| Metadata inspected | 8,240 cells. Columns: `cell`, `nGene`, `nUMI`, `orig.ident` (10 dose labels C…T320), `percent.mito`, `State` (6 values: C + S1–S5) |
| Loader adapted for this layout | New config fields `metadata`, `metadata_cell_column`, `metadata_sample_column`. Cells are joined to metadata by ID and cells without metadata are dropped (the load fails if fewer than 50% match). The count CSV is read in 1,000-row chunks into a sparse matrix so it doesn't exhaust memory |
| State task: `State` used as the state target, `"C"` ignored | Control cells have no resistance state. Leaving them out matches the paper's 5-state task, and including them would make the task partly trivial (C ≡ dose 0). Unlabelled cells are excluded from the state loss in all models |

## 8. Open items

- [x] Inspect the metadata and adapt the loader for one combined count file + metadata.
- [ ] First real-data run (download, then prepare): check that the cell IDs match between counts and metadata.
- [ ] Confirm the file layouts for GSE247690 (PDX) and GSE202435 (breast); find COV362's accession.
- [ ] First real-data run, `uv run parpi reproduce -c configs/default.yaml`, then `parpi gate`.
- [ ] Push to GitHub; set `HF_TOKEN`, `HF_MODEL_REPO`, `HF_SPACE_REPO`; tag `v0.1.0`.
