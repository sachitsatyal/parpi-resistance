"""Hierarchical pathway-attention network (needs PyTorch; runs in CI)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parpi.config import FeatureConfig, HierarchicalConfig, QCConfig  # noqa: E402
from parpi.data.datasets import resistance_target  # noqa: E402
from parpi.data.synthetic import DOSES  # noqa: E402
from parpi.models.base import Inputs, Targets  # noqa: E402
from parpi.models.hierarchical import GenePathwayLayer, HierarchicalModel  # noqa: E402
from parpi.preprocess import FeatureSpace  # noqa: E402

pytestmark = pytest.mark.torch
QC = QCConfig(min_genes=50, min_counts=300, max_counts=10**6, max_pct_mito=20, min_cells_per_gene=5)


@pytest.fixture(scope="module")
def data(synthetic):
    cd, gs = synthetic
    fs = FeatureSpace.fit(cd, gs, QC, FeatureConfig(gene_sets="synthetic"))
    _, ranks, _ = fs.transform(cd, apply_qc=False)
    dose = cd.obs["sample"].map(DOSES).to_numpy(float)
    return fs, Inputs(ranks, fs.pathway_scores(ranks)), Targets(resistance_target(dose, 320))


def test_gene_pathway_layer_starts_as_pathway_mean(data):
    fs, x, _ = data
    layer = GenePathwayLayer(fs.membership, heads=2)
    act = layer(torch.as_tensor(x.ranks - 0.5, dtype=torch.float32)).detach().numpy()
    np.testing.assert_allclose(act[:, :, 0], x.pathways, atol=1e-5)
    np.testing.assert_allclose(act[:, :, 1], x.pathways, atol=1e-5)


def test_fit_predict_attention_and_determinism(data):
    fs, x, y = data
    cfg = HierarchicalConfig(epochs=40, batch_size=64, dim=16, heads=2, patience=40, lr=3e-3)
    m1 = HierarchicalModel(cfg, fs.membership).fit(x, y, seed=0)
    m2 = HierarchicalModel(cfg, fs.membership).fit(x, y, seed=0)
    p = m1.predict(x)
    assert np.corrcoef(p.score, y.score)[0, 1] > 0.7
    np.testing.assert_allclose(p.score, m2.predict(x).score, atol=1e-5)
    att = m1.attention(x)
    np.testing.assert_allclose(att.sum(1), 1, atol=1e-5)
    assert m1.pathway_importance().shape == (len(fs.gene_sets),)
    assert m1.gene_weights().shape == (len(fs.genes), len(fs.gene_sets))
