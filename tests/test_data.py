from __future__ import annotations

import io
import tarfile

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

from parpi.celldata import CellData
from parpi.config import DatasetConfig
from parpi.data.datasets import annotate, resistance_target
from parpi.data.download import geo_suppl_url, safe_extract_tar
from parpi.data.readers import read_any, sample_name


def _cd() -> CellData:
    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 1]], dtype=np.float32))
    return CellData(
        X=X, obs=pd.DataFrame({"sample": ["T40", "C"]}), genes=np.array(["A", "B", "C"])
    )


def test_duplicate_genes_are_summed():
    X = sp.csr_matrix(np.array([[1, 2, 3]], dtype=np.float32))
    cd = CellData(X=X, obs=pd.DataFrame(index=[0]), genes=np.array(["A", "B", "A"]))
    assert list(cd.genes) == ["A", "B"]
    assert cd.X.toarray().tolist() == [[4, 2]]


def test_align_genes_pads_missing_and_reports_coverage():
    aligned, cov = _cd().align_genes(np.array(["C", "Z", "A"]))
    assert aligned.X.toarray().tolist() == [[2, 0, 1], [1, 0, 0]]
    assert cov == pytest.approx(2 / 3)


def test_concat_requires_aligned_genes():
    a = _cd()
    b, _ = _cd().align_genes(np.array(["C", "B", "A"]))
    with pytest.raises(ValueError, match="identical gene order"):
        CellData.concat([a, b])


def test_annotate_maps_conditions_and_drops_unknown():
    cfg = DatasetConfig(name="t", condition_regex=r"(C|T\d+)", dose_map={"C": 0, "T40": 40})
    cd = _cd()
    cd.obs["sample"] = ["GSM1_T40_rep1", "weird"]
    out = annotate(cd, cfg)
    assert out.n_cells == 1
    assert out.obs["dose_um"].tolist() == [40.0]


def test_resistance_target_bounds():
    t = resistance_target(np.array([0, 1, 320]), 320)
    assert t[0] == 0 and t[-1] == pytest.approx(1) and 0 < t[1] < 1


def test_sample_name():
    from pathlib import Path

    assert sample_name(Path("GSM123_T2.5_counts.csv.gz")) == "T2.5"


def test_read_mtx_dir_and_dense_csv(tmp_path):
    d = tmp_path / "T80"
    d.mkdir()
    m = sp.coo_matrix(np.array([[5, 0], [0, 7], [1, 1]]))  # genes x cells
    scipy.io.mmwrite(d / "matrix.mtx", m)
    pd.DataFrame([["e1", "GAPDH"], ["e2", "ACTB"], ["e3", "TP53"]]).to_csv(
        d / "features.tsv", sep="\t", header=False, index=False
    )
    pd.DataFrame(["AAA-1", "CCC-1"]).to_csv(d / "barcodes.tsv", sep="\t", header=False, index=False)
    cd = read_any(d)
    assert cd.X.shape == (2, 3) and list(cd.genes) == ["GAPDH", "ACTB", "TP53"]
    assert cd.obs["sample"].iloc[0] == "T80"

    dense = pd.DataFrame({"AAA": [1, 2, 3], "CCC": [0, 4, 0]}, index=["GAPDH", "ACTB", "TP53"])
    path = tmp_path / "GSM9_C_counts.csv"
    dense.to_csv(path)
    cd2 = read_any(path)
    assert cd2.X.shape == (2, 3)  # transposed to cells x genes
    assert cd2.obs["sample"].iloc[0] == "C"


def test_geo_url():
    assert geo_suppl_url("GSE206125").endswith("/GSE206nnn/GSE206125/suppl/")
    with pytest.raises(ValueError, match="accession"):
        geo_suppl_url("not-an-accession")


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        data = b"x"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError, match="unsafe"):
        safe_extract_tar(archive, tmp_path / "out")


def test_combined_matrix_with_metadata_and_states(tmp_path):
    """GSE206125 layout: one genes x cells CSV for all doses + a per-cell metadata TSV."""
    from parpi.data.datasets import load_dataset

    raw = tmp_path / "raw"
    raw.mkdir()
    cells = ["bc_ctrlI_1.1", "bc_ctrlI_2.1", "bc_T40_1.1", "bc_T40_2.1", "bc_orphan"]
    counts = pd.DataFrame(
        np.arange(15).reshape(3, 5) + 1, index=["GAPDH", "ACTB", "TP53"], columns=cells
    )
    counts.to_csv(raw / "all_counts.csv.gz")
    pd.DataFrame(
        {
            "cell": cells[:4],
            "nGene": 3,
            "orig.ident": ["C", "C", "T40", "T40"],
            "State": ["C", "C", "S2", "S3"],
        }
    ).to_csv(raw / "meta.tsv.gz", sep="\t", index=False)
    cfg = DatasetConfig(
        name="k",
        source="local",
        path=str(raw),
        files=["all_counts.csv.gz"],
        metadata="meta.tsv.gz",
        metadata_sample_column="orig.ident",
        dose_map={"C": 0, "T40": 40},
        state_column="State",
        state_ignore=["C"],
    )
    cd = load_dataset(cfg, tmp_path)
    assert cd.n_cells == 4  # orphan cell without metadata dropped
    assert sorted(cd.genes) == ["ACTB", "GAPDH", "TP53"]
    assert cd.obs["dose_um"].tolist() == [0, 0, 40, 40]
    assert cd.obs["state"].isna().tolist() == [True, True, False, False]
    row = dict(zip(cd.genes, cd.X[2].toarray().ravel(), strict=True))
    assert row == {"GAPDH": 3, "ACTB": 8, "TP53": 13}  # column 3 of the CSV
