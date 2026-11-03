# Design: from v1 to v2

v2 extends the model introduced in *Attention-Based Hierarchical Graph Autoencoder for
Dose-Specific Single-Cell Resistance Dynamics* (Satyal, Long, Gao). Its core idea is kept: a
gene → pathway → cell hierarchy with attention, treating resistance as a trajectory across doses.
v2 changes how the model is trained and tested, so the idea can be evaluated per cell, against
strong baselines, and on data it has never seen.

## Limitations of v1 and how v2 addresses them

These follow from the scope of the original, dose-level evaluation; they are not errors in the idea.

| # | v1 limitation | Why it matters | How v2 addresses it |
|---|---|---|---|
| 1 | Dose-level learning: one graph per dose, about 9 training units | Dose embeddings and extrapolation rest on few points | The unit of learning is the **cell** (thousands of samples); dose is a supervised target |
| 2 | The T640 trajectory has no data behind it | Can't be validated | Out-of-range doses are flagged `extrapolated=True`, never presented as predictions |
| 3 | The T320 result wasn't compared with a trivial baseline | Neighbouring doses are already very similar (T160→T320 drift 0.0023, cos-sim ≈ 0.998) | Every metric is reported next to "copy nearest dose", "linear interpolation" and "predict the mean" |
| 4 | Resistance labels were among the cell input features as well as the target | The model can read the answer from its input | Labels are targets only; a depth-only model checks for technical confounding |
| 5 | The autoencoder reconstructed its own embeddings | A low loss is easy to reach | The reconstruction target is real data (pathway scores) |
| 6 | DEGs and enrichment computed per dose, so node sets differed between doses | Dose differences may partly reflect different graphs | One fixed, checksummed pathway library and gene universe for all data |
| 7 | One cell line | Generalisation untested | Held-out cell lines (COV362, OVSAHO), a second platform, and PDX tumors from the same study |
| 8 | The CPA comparison used a different setup from v1 | Not like-for-like | All models share the same features, splits and metrics |

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
