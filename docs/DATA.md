# Data

All data is public and downloaded by the pipeline from GEO. Each file's SHA-256 is recorded in
`data/raw/<dataset>/MANIFEST.json`. If an upstream file changes, the pipeline stops with a
checksum error instead of silently producing different results.

| Dataset | Role | Accession | What it is |
|---|---|---|---|
| `kuramochi` | train | GSE206125 | Kuramochi (BRCA2-deficient HGSOC). Olaparib dose escalation: control, then 1 µM doubling to 320 µM. França et al., Nature 2024 |
| `pdx_ovarian` | external | GSE247690 | Ovarian PDX models from the same study (vehicle / persister / resistant) |
| `cov362` | external (disabled until its accession is confirmed) | sub-series of GSE247691 | COV362 (BRCA1-mutant HGSOC), olaparib-adapted |
| `hcc1806_breast` | external, exploratory | GSE202435 | HCC1806 TNBC, parental vs **talazoparib**-resistant |

The França SuperSeries is GSE247691. It also contains an A375/PC9 scRNA-seq series, plus
ATAC-seq, CRISPR-screen and exome data for Kuramochi.

## Before the first real run

GEO file names vary by submitter, so check them once:

```bash
parpi inspect GSE206125
parpi inspect GSE247690
parpi inspect GSE202435
```

Then adjust each dataset's `files` glob and `condition_regex` in `configs/datasets/` so that
every cell's sample name maps to a condition in `dose_map` (training) or `resistant_map`
(external). Cells whose condition is not mapped are dropped, with a warning.

- **Cell states.** If the GSE206125 release includes the published S1–S5 state per cell, set
  `state_column` to that metadata column to train the state head.
- **Breast scope.** No public breast scRNA-seq olaparib dose-escalation dataset was found
  (survey, November 2026). GSE202435 uses a different PARP inhibitor and only two conditions,
  so breast results are labelled exploratory everywhere. New breast data plugs in as one more
  YAML file in `configs/datasets/`.

## Gene sets

The default is `MSigDB_Hallmark_2020 + Reactome_2022`, fetched once from Enrichr and pinned
by SHA-256 in `data/genesets/genesets.lock.json`. The filtered library is saved inside every
model bundle (`gene_sets.gmt`), so predictions never depend on a later download.
