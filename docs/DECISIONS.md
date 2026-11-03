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
| `uv run pytest` after v3 | All tests pass on the workstation |
| Committed `uv.lock` and pushed to **github.com/sachitsatyal/parpi-resistance** (private) | CI needs the lockfile; GitHub is the source for CI/CD and Hugging Face deployment |
| Removed the `Co-Authored-By: Claude` / `Claude-Session` lines from all commit messages (`git filter-branch --msg-filter`), then force-pushed | User's choice: no AI co-author on the repository. GitHub's contributor list is cached and updates later |
| Removed `.github/dependabot.yml` | On the first push it opened a burst of dependency-update PRs. Versions are already pinned by `uv.lock`, so automatic update PRs aren't needed during development. Revisit (grouped, monthly) before a public release |
| First CI run on GitHub | lint, tests (Python 3.10 and 3.12, including PyTorch) **passed**; the Docker job **failed** |
| Dockerfile fixes: uv image 0.5 -> 0.13; `--no-editable` install; venv kept at the same path (`/app/.venv`) in builder and runtime | uv 0.5 can't read a lockfile written by uv 0.13. An editable install points at `/app/src`, which isn't in the runtime image. Venv scripts hard-code their build path |
| CI after the Docker fix | **All jobs green** (lint, tests 3.10/3.12, docker) |
| First `prepare` on Kuramochi | Download OK. The count matrix has **18,209 cells** but the metadata has **8,240**, and only **7,322** match. The ID format is identical, so this is about cell sets, not naming. Stopped by the 50% match guard. Diagnosing with `scripts/check_ids.py` before choosing which cells to use |
| Cell-ID diagnosis (`scripts/check_ids.py`) | Every dose except T40 has about 50% of its matrix cells in the metadata (an authors' filter, likely G1-only per the `_g1` file name). **T40 metadata uses a second library (`bc_40uM2nd_`, 918 cells) that isn't in the matrix**, while the matrix's T40 library (`bc_40uM_`, 2,077 cells) has no metadata. States are cleanly ordered: C → S1 (T1–T5) → S2–S4 (T2.5–T80) → S5 (T160–T320) |
| **Decision: keep all matrix cells, take dose from the barcode prefix, apply uniform QC to every dose; published states used as labels where available** (`metadata_required: false`, `barcode_sample_regex`) | Metadata-only matching would drop T40 entirely, and T40 is the S3→S4 transition. Using curated cells for some doses but uncurated cells for T40 would confound dose with processing. Uniform QC on all ~18k cells avoids that and doubles the data. `orig.ident` is kept only to cross-check doses (the load fails if any disagree). Per-dose QC medians are written to `summary.json` to catch an outlier library. Alternative kept available: `metadata_required: true` (authors' curated cells, no T40) |
| `load_config` reads YAML as UTF-8 explicitly | Windows defaults to cp1252, which can mis-read non-ASCII characters in configs |
| `prepare` with all cells | 18,209 cells, **QC removed none**. Feature space: 7,350 genes, 1,266 gene sets. Barcode doses agree with metadata for all 7,322 matched cells |
| QC diagnosis (`scripts/check_qc.py`) | The matrix is the authors' raw counts: our counts equal metadata `nUMI` and `nGene` exactly (ratio 1.0, r = 1.0). (An earlier worry about "17k UMIs per cell" was wrong: it came from the first 5 rows, not the median.) Cells without metadata have **lower depth in most doses** (control median 1,670 vs 5,946 UMIs; similar depth at 10, 320 µM). So the authors' filter is partly quality-based, not only cell-cycle |
| **Decision: keep all cells, but plan a sensitivity analysis** on curated cells only (`metadata_required: true`, which drops T40) | Rank-based features are depth-invariant (tested) and the depth-only sanity model detects depth confounding. Uniform processing keeps T40. Comparing the two CV results tells us whether the uncurated cells change the conclusions |
| First `train` (ensemble 5, network: 60 epochs, 4 heads, dim 64) | Ridge and HGB finished (HGB 835 s). HGB: cell MAE 0.187 vs 0.295 mean baseline, dose-level Spearman 0.77 (just under the 0.8 gate), state balanced accuracy 0.44 vs chance 0.43. **Network stage >45 min with no progress output → stopped** |
| Lighter run: ensemble 3, network 25 epochs / patience 4 / 2 heads / dim 32 / batch 512; per-fold progress logging added | Make a CPU run finish in reasonable time. Silent long stages made it impossible to tell whether the run was progressing |

## 8. Open items

- [x] Inspect the metadata and adapt the loader for one combined count file + metadata.
- [x] Cell IDs diagnosed and the cell-selection decision made.
- [x] Rerun `prepare`; check per-dose QC (T40 looks like its neighbours: median 2,097 UMIs).
- [ ] Train and cross-validate on all cells.
- [ ] Sensitivity analysis: curated cells only.
- [ ] Confirm the file layouts for GSE247690 (PDX) and GSE202435 (breast); find COV362's accession.
- [ ] First real-data run, `uv run parpi reproduce -c configs/default.yaml`, then `parpi gate`.
- [x] Push to GitHub.
- [ ] Set `HF_TOKEN`, `HF_MODEL_REPO`, `HF_SPACE_REPO`; tag `v0.1.0`.
