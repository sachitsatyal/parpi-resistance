# Design: what was wrong before, and what this rebuild does instead

This project replaces the hierarchical graph autoencoder described in *Attention-Based
Hierarchical Graph Autoencoder for Dose-Specific Single-Cell Resistance Dynamics*. Its core
idea is kept: a gene → pathway → cell hierarchy with attention. The problems below are why
the rest was redesigned rather than refactored.

## Problems in the original and the fix for each

| # | Problem in the original | Consequence | Fix here |
|---|---|---|---|
| 1 | One graph per dose, so the dose-level model learns from **9 samples** | The LSTM, dose embeddings and "extrapolation" are fitted to 9 points | The unit of learning is the **cell** (thousands of samples). Dose is a supervised target, not a learned slot |
| 2 | T640 "extrapolation" appends an untrained embedding slot | Nothing supports the T640 prediction | Out-of-range doses are flagged `extrapolated=True` and never presented as validated |
| 3 | Reported T320 prediction (cos-sim 0.983) is **worse than copying T160** (real T160→T320 cos-distance 0.0023 → cos-sim ≈ 0.998) | The headline result loses to a trivial predictor | Every metric is reported against "copy nearest dose", "linear interpolation" and "predict the mean" |
| 4 | Resistance labels (5 of 7 cell features) are **inputs** and also the **target** | Label leakage; the reported 54% accuracy is hard to interpret | Labels are only targets. A depth-only model checks for technical confounding |
| 5 | Autoencoder reconstructs **its own embeddings** | Low loss is close to guaranteed and not evidence of biology | The reconstruction target is real data (pathway scores) |
| 6 | DEGs and enrichment re-run **per dose**, so every graph has different genes and pathways | Dose clusters in t-SNE partly reflect different node sets | One fixed, checksummed pathway library and gene universe for all data |
| 7 | Only one cell line | No evidence of generalisation | External validation on other cell lines and PDX samples from the same study, plus an exploratory breast line |
| 8 | CPA comparison 0.98 vs 0.13 | Implies mismatched evaluation, not a better model | Baselines share the same features, splits and metrics |

## Features: per-cell rank scores

Pathway activity is the mean **within-cell rank** of the pathway's genes (singscore-style).
Ranks need no dataset-level statistics, which gives three properties:

- A user can score one sample without a matched control.
- Sequencing depth cancels. This is tested: scaling all counts by 3 leaves the scores
  unchanged.
- Training and inference use the exact same transform.

The hierarchical network starts from the same ranks. Its gene → pathway layer is initialised
to a uniform average, so at step 0 it reproduces the baseline features. It can only improve
on them by learning which genes in each pathway matter.

## Targets

- **Resistance score:** `log2(dose+1) / log2(max_dose+1)`, the cell's position along the
  escalation. It is defined for every training cell and doesn't depend on any clustering.
- **Cell state (optional):** the published S1–S5 states, if available. These are used as a
  supervised target only and scored with balanced accuracy, macro-F1 and calibration error
  (ECE), alongside chance and majority-class rates.

## Evaluation protocol

1. **Leave-dose-out CV.** Conditions are sorted by dose and dealt round-robin into 5 folds.
   Every fold mixes low and high doses, so the CV covers both interpolation and the two
   extremes. No dose appears in both train and test.
2. **Fair baselines.** The "predict the mean" baseline is computed out-of-fold with the same
   folds. Under leave-dose-out CV, pooled correlations of a near-constant predictor only
   reflect per-fold intercepts. That's why the depth-only check is judged by error, not by
   correlation.
3. **External validation.** Resistant-vs-parental AUROC on datasets never used for training
   or model selection.
4. **Dose-response.** Each interior dose is held out and the curve is compared with
   copy-nearest-dose and straight-line interpolation. In development, a global spline lost
   to linear interpolation; that result is why PCHIP is used.
5. **Model selection.** Highest dose-level Spearman, then lower MAE, then the simpler model.
   The hierarchical network ships only if it wins.
6. **Release gate.** Dose-level Spearman ≥ 0.8. Cell MAE ≤ 0.8 × the mean baseline. AUROC
   ≥ 0.65 on every external dataset of the training cancer type. Breast results are
   reported but not gated.
