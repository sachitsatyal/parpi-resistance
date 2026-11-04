"""The released artifact: feature space + fitted model + metadata, and the Predictor API.

A bundle is a plain folder, so it can live on disk, in a GitHub release or on
the Hugging Face Hub (``ModelBundle.load("hf://org/repo")``).
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from parpi import __version__
from parpi.celldata import CellData
from parpi.models.base import Inputs, Prediction
from parpi.models.dose_response import DoseResponseModel
from parpi.preprocess import FeatureSpace

REACTOME = re.compile(r"R-HSA-\d+\s*$")
BUNDLE_FORMAT = 1


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_info() -> dict[str, str]:
    import scipy
    import sklearn

    info = {
        "python": platform.python_version(),
        "parpi": __version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


@dataclass
class ModelBundle:
    features: FeatureSpace
    model: Any  # Ensemble
    dose_response: DoseResponseModel | None
    metadata: dict[str, Any] = field(default_factory=dict)
    folder: Path | None = None  # where the bundle was loaded from (model card lives there)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.features.save(folder)
        joblib.dump(self.model, folder / "model.joblib", compress=3)
        if self.dose_response is not None:
            joblib.dump(self.dose_response, folder / "dose_response.joblib", compress=3)
        meta = {
            "bundle_format": BUNDLE_FORMAT,
            "git_commit": git_commit(),
            "environment": environment_info(),
            **self.metadata,
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, location: str | Path) -> ModelBundle:
        folder = _resolve(location)
        meta = json.loads((folder / "metadata.json").read_text())
        if meta.get("bundle_format") != BUNDLE_FORMAT:
            raise ValueError(f"unsupported bundle format {meta.get('bundle_format')}")
        dr_path = folder / "dose_response.joblib"
        return cls(
            features=FeatureSpace.load(folder),
            model=joblib.load(folder / "model.joblib"),
            dose_response=joblib.load(dr_path) if dr_path.exists() else None,
            metadata=meta,
            folder=folder,
        )

    # ----------------------------------------------------------- inference
    def featurize(self, cd: CellData, apply_qc: bool = True) -> tuple[CellData, Inputs, float]:
        qc_cd, ranks, coverage = self.features.transform(cd, apply_qc=apply_qc)
        return qc_cd, Inputs(ranks, self.features.pathway_scores(ranks)), coverage

    def predict(
        self, cd: CellData, apply_qc: bool = True, top_k: int = 5, explain: str = "hallmark"
    ) -> pd.DataFrame:
        """Per-cell predictions with uncertainty and top contributing pathways.

        ``explain="hallmark"`` (default) names only Hallmark pathways in ``top_pathways``;
        ``"all"`` also allows Reactome sets, whose names can mislead (see `_explainable`).
        """
        qc_cd, x, coverage = self.featurize(cd, apply_qc=apply_qc)
        pred: Prediction = self.model.predict(x)
        out = qc_cd.obs[[c for c in ("barcode", "sample") if c in qc_cd.obs]].copy()
        out["resistance_score"] = pred.score
        out["resistance_score_std"] = pred.score_std
        if pred.state_proba is not None and pred.state_classes is not None:
            for j, c in enumerate(pred.state_classes):
                out[f"p_state_{c}"] = pred.state_proba[:, j]
            out["predicted_state"] = pred.state_classes[pred.state_proba.argmax(1)]
        out["top_pathways"] = self._top_pathways(x, top_k, explain)
        out.attrs["gene_coverage"] = coverage
        return out.reset_index(drop=True)

    def _signed_weights(self) -> np.ndarray | None:
        """Mean standardised coefficient per pathway for linear ensembles, else None."""
        coefs = []
        for member in getattr(self.model, "members", []):
            steps = getattr(getattr(member, "reg", None), "steps", None)
            last = steps[-1][1] if steps else None
            if last is None or not hasattr(last, "coef_"):
                return None
            coefs.append(np.ravel(last.coef_))
        return np.mean(coefs, axis=0) if coefs else None

    def _explainable(self, explain: str) -> np.ndarray:
        """Columns allowed in explanations.

        Reactome sets (names ending in ``R-HSA-<id>``) overlap heavily: in the release
        model, "Downstream TCR Signaling", "Activation Of NF-kappaB In B Cells",
        "Programmed Cell Death" and "Vpu Mediated Degradation Of CD4" each contain the
        same 43 proteasome genes, so immune and viral names appear for ovarian cells.
        Hallmark sets have fewer, clearer names. The model still uses every set; only
        the explanation is restricted. Falls back to all sets if none are Hallmark.
        """
        names = self.features.gene_sets.names
        if explain == "all":
            return np.arange(len(names))
        if explain != "hallmark":
            raise ValueError(f"explain must be 'hallmark' or 'all', not {explain!r}")
        keep = [i for i, n in enumerate(names) if not REACTOME.search(n)]
        return np.array(keep) if keep else np.arange(len(names))

    def _top_pathways(self, x: Inputs, k: int, explain: str = "hallmark") -> list[str]:
        """The pathways that move each cell's score the most, in either direction.

        Linear models: contribution = standardised weight x the cell's standardised
        pathway activity (vs. training cells). Pathways are ranked by the size of the
        contribution and tagged with the cell's activity ("high"/"low" vs. training
        cells) and its effect ("raises score"/"lowers score"), e.g.
        ``MYC targets (high, raises score)``. Other models have no signed weights, so
        names are ranked by importance x |activity| and tagged "high"/"low" only.
        """
        names = np.array(self.features.gene_sets.names)
        # Standardise against the *training* distribution so results do not depend
        # on what else is in the query batch.
        mu = np.asarray(self.metadata.get("pathway_mean", x.pathways.mean(0)))
        sd = np.asarray(self.metadata.get("pathway_std", x.pathways.std(0)))
        z = (x.pathways - mu) / (sd + 1e-9)
        w = self._signed_weights()
        contrib = z * (w if w is not None else self.model.pathway_importance())
        allowed = self._explainable(explain)
        order = np.argsort(-np.abs(contrib[:, allowed]), axis=1)[:, :k]
        top = allowed[order]

        def tag(i: int, j: int) -> str:
            level = "high" if z[i, j] >= 0 else "low"
            if w is None:
                return level
            return f"{level}, {'raises' if contrib[i, j] >= 0 else 'lowers'} score"

        return [";".join(f"{names[j]} ({tag(i, j)})" for j in row) for i, row in enumerate(top)]

    def pathway_importance(self) -> pd.DataFrame:
        imp = self.model.pathway_importance()
        return pd.DataFrame(
            {"pathway": self.features.gene_sets.names, "importance": imp}
        ).sort_values("importance", ascending=False, ignore_index=True)

    def dose_curve(self, doses_um: np.ndarray, pathways: list[str] | None = None) -> pd.DataFrame:
        if self.dose_response is None:
            raise ValueError("this bundle has no dose-response model")
        prof, extrap = self.dose_response.predict(np.asarray(doses_um, dtype=float))
        names = list(self.features.gene_sets.names)
        df = pd.DataFrame(prof, columns=names)
        if pathways:
            df = df[pathways]
        df.insert(0, "dose_um", doses_um)
        df.insert(1, "extrapolated", extrap)
        return df


def _resolve(location: str | Path) -> Path:
    loc = str(location)
    if loc.startswith("hf://"):
        from huggingface_hub import snapshot_download

        repo, _, rev = loc[5:].partition("@")
        return Path(snapshot_download(repo_id=repo, revision=rev or None))
    return Path(loc)
