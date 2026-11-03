from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from parpi.celldata import CellData
from parpi.config import FeatureConfig, QCConfig
from parpi.genesets import parse_gmt
from parpi.preprocess import FeatureSpace, filter_cells, qc_metrics, rank_normalize

QC = QCConfig(min_genes=50, min_counts=300, max_counts=10**6, max_pct_mito=20, min_cells_per_gene=5)
FEATS = FeatureConfig(gene_sets="synthetic", min_set_size=10, max_set_size=300)


def test_parse_gmt_and_membership():
    gs = parse_gmt("S1\tdesc\tA\tB\tC\nS2\t\tB,1.0\tD\n", "x")
    assert gs.names == ("S1", "S2") and gs.members[1] == ("B", "D")
    f = gs.filter(np.array(["A", "B", "C"]), min_size=2, max_size=5)
    assert f.names == ("S1",)
    M = gs.membership(np.array(["A", "B", "C", "D"])).toarray()
    assert M.sum(0).tolist() == [3, 2]


def test_qc_filters_low_quality_cells(synthetic):
    cd, _ = synthetic
    X = cd.X.tolil()
    X[0, :] = 0  # empty cell
    bad = CellData(X=X.tocsr(), obs=cd.obs.copy(), genes=cd.genes)
    out = filter_cells(bad, QC)
    assert out.n_cells == cd.n_cells - 1
    assert {"total_counts", "n_genes", "pct_mito"} <= set(out.obs.columns)
    assert qc_metrics(cd)["pct_mito"].between(0, 10).all()


def test_rank_normalize_range_and_order():
    X = sp.csr_matrix(np.array([[0, 5, 1], [3, 0, 0]], dtype=np.float32))
    R = rank_normalize(X)
    assert R.min() > 0 and R.max() < 1
    assert R[0].argsort().tolist() == [0, 2, 1]


def test_pathway_scores_invariant_to_sequencing_depth(synthetic):
    cd, gs = synthetic
    fs = FeatureSpace.fit(cd, gs, QC, FEATS)
    _, r1, _ = fs.transform(cd, apply_qc=False)
    deeper = CellData(X=cd.X * 3, obs=cd.obs, genes=cd.genes)
    _, r3, _ = fs.transform(deeper, apply_qc=False)
    np.testing.assert_allclose(fs.pathway_scores(r1), fs.pathway_scores(r3), atol=1e-6)


def test_low_gene_coverage_is_refused(synthetic):
    cd, gs = synthetic
    fs = FeatureSpace.fit(cd, gs, QC, FEATS)
    few = cd.subset(genes=np.arange(50))
    with pytest.raises(ValueError, match="genes are present"):
        fs.transform(few, apply_qc=False)


def test_feature_space_roundtrip(tmp_path, synthetic):
    cd, gs = synthetic
    fs = FeatureSpace.fit(cd, gs, QC, FEATS)
    fs.save(tmp_path)
    fs2 = FeatureSpace.load(tmp_path)
    np.testing.assert_array_equal(fs.genes, fs2.genes)
    assert fs.gene_sets.names == fs2.gene_sets.names
    assert (fs.membership != fs2.membership).nnz == 0


def test_exclude_gene_sets_by_name():
    gs = parse_gmt(
        "Peptide Chain Elongation\t\tA\tB\n"
        "Post-translational Protein Modification\t\tA\tC\n"
        "Eukaryotic Translation Initiation\t\tB\tC\n"
        "DNA Repair\t\tA\tB\tC\n",
        "x",
    )
    kept = gs.exclude(["peptide chain elongation", r"\btranslation\b"])
    assert kept.names == ("Post-translational Protein Modification", "DNA Repair")
    with pytest.raises(ValueError, match="every gene set"):
        gs.exclude([".*"])
