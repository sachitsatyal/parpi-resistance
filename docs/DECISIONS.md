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
| **First complete real-data CV** (18,209 cells, leave-dose-out, 5 folds; config fingerprint `32b8cc641f58`) | Cell MAE (mean baseline 0.295): **hierarchical 0.120**, ridge 0.154, HGB 0.187, depth-only 0.302. Dose-level Spearman: hierarchical 0.867 = ridge 0.867 > HGB 0.770. State balanced accuracy (chance 0.433): **hierarchical 0.665**, ridge 0.633, HGB 0.436. State ECE: hierarchical **0.104**, ridge 0.179, HGB 0.322. Runtime: ridge 19 s, HGB 498 s, network 1,327 s; the whole run took about 51 min |
| **Selected: hierarchical** (tied on dose Spearman → lower MAE wins) | The pathway-attention network earns its place on honest held-out-dose testing: 22% lower error than ridge, the best state accuracy and the best calibration |
| Interpretation notes | Depth-only (0.302) is no better than predicting the mean, so **sequencing depth carries no dose signal**, even though median depth differs between doses. HGB underperforms mainly because trees can't extrapolate to held-out extreme doses (0 and 320 µM folds). Dose Spearman 0.867 over 10 doses means a couple of neighbouring doses are mis-ordered by both models; inspect per-dose predictions next |
| Release gate on Kuramochi-only config | **Passed** |
| Per-dose held-out predictions (`scripts/per_dose.py`), hierarchical | **≥10 µM tracks the continuum**: 0.45, 0.44, 0.62, 0.68, 0.88, 0.87 for 10–320 µM. The only swaps are near-ties: 10 vs 20 µM (0.450 vs 0.443) and 160 vs 320 µM (0.879 vs 0.868; both doses are 100% S5, so biologically one state). **Low doses collapse**: C 0.31, T1 0.24, T2.5 0.20, T5 0.22, so the control is ranked 4th. Ridge shows the same pattern |
| Explanation of the low-dose collapse (two causes) | (1) **Extrapolation in CV**: the fold holding out the control has no 0 µM cells to learn from, and the extreme folds pull toward the middle (320 µM is under-predicted, 0.87). The deployed model is trained on the control, so this tests a harder case than deployment. (2) **Biology**: T1–T5 are mostly one state (S1), so a smooth dose target over-asks the transcriptome. This fits the paper's stepwise "resistance continuum" |
| Decision: **keep leave-dose-out as the primary metric** (no change after seeing results); test the control question on independent untreated samples (PDX vehicle, COV362 parental) in external validation | Changing the evaluation after seeing results would bias it. Whether an untreated sample scores low is best answered on data the model has never seen |
| Uncertainty is under-estimated | Mean ensemble std 0.02–0.06 vs actual MAE 0.12: the bootstrap spread misses dose-extrapolation error. **To do:** calibrate the score intervals on CV residuals (conformal-style) before release |
| PDX layout (`scripts/check_pdx.py`, GSE247690) | Two experiments, each with a combined count matrix + curated metadata; IDs `BARCODE_SAMPLE`, all unique. **Resistance**: V1, V2 (vehicle); R1.1, R1.2, R3.1, R3.2 (relapsed on olaparib); R3.V (R3 line on vehicle). 26,004 matrix cells, 10,918 curated (10,917 found). **Persistence**: V1, V2; D1.1–D4.2 persisters; 8,756 cells, 4,843 curated |
| **External test design** (`configs/ovarian.yaml`) | Primary, gated: `pdx_resistance` on the **authors' curated cells**, AUROC vehicle vs R1+R3. Report-only: `pdx_resistance_allcells` (every cell) and `pdx_persistence` (mean score per sample). R3.V is scored but excluded from the AUROC |
| Why curated cells for external but all cells for training | PDX tumors contain mouse stromal cells, and the authors' independent filter is the only available way to remove them. The filter wasn't tuned by us, so it doesn't bias the test. Robustness to curation is shown by the all-cells variant |
| New `gate:` dataset flag | Lets ambiguous or secondary sets be reported without blocking a release |
| Metadata match check changed | Now checks the fraction of **metadata** cells found in the counts (an ID-format problem), not the fraction of matrix cells with metadata: curated subsets are legitimately small (PDX 42%) |
| Evaluating with `ovarian.yaml` reuses the models trained with `kuramochi.yaml` | The training data, features and `train:` settings are identical (enforced by a test), so retraining for 50 min would reproduce the same models |
| PDX `prepare` | Gene coverage 97.7% (resistance), 99.97% (persistence). 100% of the curated resistance cells found in the counts. QC kept 10,673 / 10,918 curated cells |
| **External PDX result: the score does not transfer** | Cell-level AUROC, vehicle vs R1+R3 (curated): hierarchical **0.511**, ridge 0.531, HGB 0.503; all-cells variant 0.49–0.54. Mean hierarchical score: V1 0.54, **V2 0.64**, resistant 0.58–0.63, R3.V 0.58. **The two vehicle tumors differ more from each other than vehicle does from resistant.** The release gate (≥0.65) would fail, and it should |
| Persistence (report-only) | Hierarchical: 7 of 8 persister samples score above both vehicles (0.58–0.69 vs 0.54–0.57). This hints that the score tracks **acute drug exposure / stress** more than stable resistance; only 2 vehicles, so weak evidence |
| Likely causes | (1) **Domain shift**: in-vitro cell line → in-vivo PDX from a different patient; the whole PDX distribution sits mid-scale (~0.55–0.65). (2) **Platform shift**: inDrop (Kuramochi) → 10x (PDX). (3) **Sample is the real unit**: 2 vehicle vs 4 resistant tumors, so cell-level AUROC is dominated by tumor-to-tumor differences. (4) The Kuramochi score may partly encode cell-line-specific or acute-dose features |
| **Decision: keep the gate and the result as they are; no release yet.** Next: a pathway-level concordance test (`scripts/concordance.py`) with a vehicle-vs-vehicle NULL, to see whether the biology transfers even though the trained weights don't | This decides between the two remedies: (a) reference-normalised scoring against a matched control sample, or (b) training on more than one model system (COV362 + Kuramochi), validated on PDX |
| **Pathway concordance Kuramochi ↔ PDX** (`scripts/concordance.py`, 1,266 pathways, Spearman of standardised effects) | Resistant vs vehicle: **−0.07** (late doses), +0.00 (early). Persisters vs vehicle: −0.02 / +0.13. R3.V: +0.02 / +0.07. **NULL, vehicle V2 vs V1 (resistance exp.): −0.62 / −0.69**; persistence-exp. NULL ≈ 0. → The resistance biology learned in Kuramochi **does not reappear in PDX**, even at the pathway level and relative to vehicle |
| What the Kuramochi dose signal is | The top late-dose pathways are almost all **down-regulated ribosome/translation/NMD, MYC targets, amino-acid metabolism, HSF1**: a generic biosynthesis/proliferation slowdown. That same axis is the **largest difference between the two untreated PDX tumors** (V2 vs V1 ≈ +0.7–1.0 on those pathways). So the dominant dose axis isn't resistance-specific, which explains why the model scores V2 high |
| Consequences | (1) **Reference normalisation won't fix it**: the concordance contrasts are already relative to vehicle and still ≈ 0. (2) The current model is valid **within the Kuramochi system** (it tracks dose adaptation) but must not be presented as a general resistance score. (3) With 6 tumors, any PDX test is low-powered |
| **Decision: pre-registered next protocol; PDX is used only once more** | To keep PDX an honest test set: (a) add a second olaparib-adapted ovarian line (**COV362**) to training; (b) choose any removal of generic axes (proliferation, translation, cell cycle) by **leave-cell-line-out CV on the training lines only**; (c) then one final PDX evaluation, reported whatever the outcome. Each attempt is logged here |
| More França data found (`scripts/check_more_lines.py`) | **GSE247689: COV362 and OVSAHO**, each C, T5, T10, T20, T40 (5,535 / 8,018 matrix cells; 2,325 / 3,638 curated), the same layout as Kuramochi, no state labels. **GSE247688: Kuramochi on 10x**: C, T10.A, T320.A, T10.P, T320.P × 2 replicates (13,800 cells) |
| **Pre-registered multi-line round** (`configs/multiline_a.yaml`, `configs/multiline_b.yaml`; recorded before any results) | Train on **Kuramochi + COV362 + OVSAHO**: all cells, uniform QC, dose from the barcode, **one shared absolute dose scale** (log µM, 0–320). Selection by **leave-one-cell-line-out CV** (mean over held-out lines of dose-level Spearman, then MAE). Variant **A** = all pathways; variant **B** = without generic biosynthesis/proliferation sets (ribosome, `\btranslation\b`, peptide-chain elongation, NMD, selenoamino acid, SRP-dependent, rRNA, influenza, SARS-CoV, MYC targets, E2F targets, G2-M checkpoint, mitotic). DNA replication and repair are kept. Models: ridge + hierarchical (HGB dropped: it can't extrapolate). Kuramochi-10x is report-only, a direct platform-shift check (AUROC C vs T320.A; ".P" meaning unconfirmed). **PDX excluded from both configs** (a test enforces this) |
| Why one shared dose scale | Per-line scaling would assume each line's top dose (40 vs 320 µM) means "fully resistant". The absolute concentration a cell tolerates is the more defensible common unit |
| Code for this round | `split.strategy: leave_dataset_out` with per-line metrics (dose Spearman, MAE, AUROC control vs top dose); `features.exclude_gene_sets` (regex on names); `processed_dir` so variants don't overwrite each other; QC summary keyed `dataset/condition` |
| Multi-line `prepare` (variant A) | 31,762 training cells (Kuramochi 18,209, COV362 5,535, OVSAHO 8,018); QC removed none; feature space **8,853 genes, 1,366 gene sets**; Kuramochi-10x gene coverage 99.98%. All metadata cells found (COV362, OVSAHO 100%); barcode doses agree with metadata everywhere. Depth differs by platform: inDrop Kuramochi ~2k UMIs vs 10x COV362/OVSAHO ~6–11k (rank features are depth-invariant) |
| Bug: `train` crashed in the state metrics | A held-out line with no state labels (COV362/OVSAHO) gave an empty fold. Folds without labelled cells are now skipped. Regression test added (fails without the fix, passes with it) |
| Rerun failed: GEO listing timed out (`ftp.ncbi.nlm.nih.gov`, 60 s read timeout) although every file was already downloaded | `fetch_geo` always asked GEO for its file list. Fix: (1) **offline path**: if every requested pattern is matched by checksummed local files in `MANIFEST.json`, no network call is made, and checksums are still verified. (2) **Retries with backoff** (5 attempts, 2–32 s) for listing and downloads. Test: the cached path must not touch the network, and a tampered cached file still fails the checksum |
| **Variant A (all pathways), leave-one-line-out CV** | Per held-out line: dose Spearman / cell MAE (mean baseline) / AUROC control vs top dose. **Ridge**: COV362 0.70 / 0.201 (0.226) / 0.756; **Kuramochi 0.99 / 0.247 (0.307) / 0.942**; OVSAHO 0.90 / 0.203 (0.191) / 0.778. **Hierarchical**: COV362 0.70 / 0.196 / 0.760; Kuramochi 0.99 / 0.244 / 0.944; OVSAHO 0.90 / 0.212 / 0.810. Mean dose Spearman 0.863 for both; mean MAE 0.2168 vs 0.2175 → tie at 3 decimals → **ridge selected** (the simpler model wins ties). Hierarchical CV took 1,711 s |
| Interpretation (A) | **The ordering transfers across cell lines in vitro**: trained only on COV362 + OVSAHO (10x, ≤40 µM), the model ranks Kuramochi's doses (inDrop, up to 320 µM) almost perfectly (0.99, AUROC 0.94). So a shared adaptation signal exists, and the platform shift alone doesn't destroy it. **Absolute levels don't transfer**: MAE barely beats the mean baseline (worse for OVSAHO), so the score is a *ranking within a system*, not a calibrated absolute value |
| **Variant B (generic growth programmes removed)**, leave-one-line-out CV | 1,297 gene sets (69 removed). Ridge: mean dose Spearman **0.859**, MAE 0.2173 (COV362 0.70 / AUROC 0.765; Kuramochi 0.976 / 0.933; OVSAHO 0.90 / 0.780). Hierarchical: 0.855, MAE 0.2230. Removing the growth programmes did **not** help: the shared cross-line signal isn't just proliferation |
| **Pre-registered selection: variant A, ridge** | Variant A has the higher mean per-line dose Spearman (0.863 vs 0.859). Within A, ridge and hierarchical tie, so the simpler model wins |
| **Kuramochi-10x (platform check, report-only)**, variant A ridge | **AUROC C vs T320.A = 0.9999.** Mean score: C 0.09, T10.A 0.44, T320.A 0.70; **T10.P 0.18–0.21, T320.P 0.36**. Hierarchical is the same (0.99996). → **Platform shift is ruled out** as the cause of the PDX failure: the same line on 10x is separated almost perfectly. The ".P" samples score far below ".A" at the same dose but above control, consistent with P = drug withdrawn (partial reversal with some retained adaptation, as in França's epigenetically reinforced states). P's meaning is still unconfirmed |
| Caveat | Kuramochi-10x is the *same cell line* as the training Kuramochi, so it tests platform robustness, not new biology |
| Final PDX evaluation set up (`configs/final_pdx.yaml`) | Identical to `multiline_a.yaml` plus the PDX sets (a test enforces this). It reuses the trained variant-A models. **Run once; the result is reported whatever it is** |

## 8. Open items

- [x] Inspect the metadata and adapt the loader for one combined count file + metadata.
- [x] Cell IDs diagnosed and the cell-selection decision made.
- [x] Rerun `prepare`; check per-dose QC (T40 looks like its neighbours: median 2,097 UMIs).
- [x] Train and cross-validate on all cells (hierarchical selected).
- [x] Per-dose diagnostics (control and T1–T5 collapse; ≥10 µM ordered).
- [ ] Calibrate score uncertainty on CV residuals.
- [ ] Sensitivity analysis: curated cells only.
- [x] Confirm the GSE247690 (PDX) layout.
- [x] Run external PDX evaluation (**does not transfer**: AUROC ≈ 0.51).
- [x] Pathway concordance Kuramochi ↔ PDX (**no transfer**; dose axis dominated by translation/MYC).
- [x] Find COV362 (and OVSAHO, Kuramochi-10x): GSE247689, GSE247688.
- [x] Run multi-line variants A and B (**A selected, ridge**; platform shift ruled out).
- [ ] Single final PDX evaluation of the selected variant.
- [ ] Confirm the GSE202435 (breast) layout; find COV362's accession.
- [ ] First real-data run, `uv run parpi reproduce -c configs/default.yaml`, then `parpi gate`.
- [x] Push to GitHub.
- [ ] Set `HF_TOKEN`, `HF_MODEL_REPO`, `HF_SPACE_REPO`; tag `v0.1.0`.

## 9. Road map (agreed 2026-11-03)

1. Finish CV and model selection (done).
2. Sensitivity analysis on the authors' curated cells only.
3. External validation: ovarian PDX (GSE247690) and COV362.
4. Exploratory breast check: HCC1806 parental vs talazoparib-resistant (GSE202435).
5. Biology check: do the top pathways match França et al. (NRF2/ATF4 stress, metabolic reprogramming, EMT as an enabler)?
6. Dose-response curves and calibration of the uncertainty estimates.
7. Release: model on the HF Hub, app on an HF Space; **research use only**.
8. Patient data: search for public pre/post-PARPi patient scRNA-seq. If found, it becomes the key test of whether the score transfers to patients. Clinical use would need prospective validation and regulatory review, which is out of scope.
