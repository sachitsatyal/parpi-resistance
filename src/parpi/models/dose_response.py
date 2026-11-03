"""Continuous dose-response curves for pathway activity.

Dose enters as a continuous variable, log2(µM + 1). Per-dose mean pathway
profiles are joined by shape-preserving piecewise-cubic (PCHIP) interpolation:
exact at measured doses, smooth in between, and free of the overshoot that
global splines show around sharp transitions (such as T40 -> T80).

Uncertainty comes from bootstrapping cells within each dose. Outside the
measured range the curve is held at the boundary value and flagged as
extrapolated; such values are not presented as validated predictions.

The model is benchmarked by leaving each interior dose out against two trivial
predictors: copy the nearest measured dose, and straight-line interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def log_dose(dose_um: np.ndarray) -> np.ndarray:
    return np.log2(np.asarray(dose_um, dtype=float) + 1.0)


def _dose_means(d: np.ndarray, pathways: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(pathways)
    df["d"] = d
    means = df.groupby("d").mean()
    return means.index.to_numpy(), means.to_numpy()


@dataclass
class DoseResponseModel:
    n_boot: int = 50
    seed: int = 0

    def fit(self, dose_um: np.ndarray, pathways: np.ndarray) -> DoseResponseModel:
        d = log_dose(dose_um)
        self.doses_, self.means_ = _dose_means(d, pathways)
        if len(self.doses_) < 2:
            raise ValueError("dose-response needs at least two distinct doses")
        self.interp_ = PchipInterpolator(self.doses_, self.means_, axis=0, extrapolate=False)
        # Bootstrap cells within each dose for confidence bands.
        rng = np.random.default_rng(self.seed)
        groups = [np.where(d == lv)[0] for lv in self.doses_]
        boots = []
        for _ in range(self.n_boot):
            m = np.stack([pathways[rng.choice(g, len(g))].mean(0) for g in groups])
            boots.append(m)
        self.boot_means_ = np.stack(boots)  # (B, n_doses, K)
        return self

    def _eval(self, interp: PchipInterpolator, means: np.ndarray, d: np.ndarray) -> np.ndarray:
        dc = np.clip(d, self.doses_.min(), self.doses_.max())
        return np.asarray(interp(dc))

    def predict(self, dose_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (pathway profile per dose, boolean 'extrapolated' flag per dose)."""
        d = log_dose(dose_um)
        extrap = (d < self.doses_.min() - 1e-9) | (d > self.doses_.max() + 1e-9)
        return self._eval(self.interp_, self.means_, d), extrap

    def predict_interval(
        self, dose_um: np.ndarray, level: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap (lower, upper) bands, shape (n_doses, K)."""
        d = log_dose(dose_um)
        curves = np.stack(
            [
                self._eval(PchipInterpolator(self.doses_, b, axis=0, extrapolate=False), b, d)
                for b in self.boot_means_
            ]
        )
        a = (1 - level) / 2
        return np.quantile(curves, a, axis=0), np.quantile(curves, 1 - a, axis=0)


def leave_one_dose_out(dose_um: np.ndarray, pathways: np.ndarray) -> pd.DataFrame:
    """Interpolation benchmark. Only interior doses are held out (extrapolation is not scored)."""
    d = log_dose(dose_um)
    levels = np.unique(d)
    rows = []
    for held in levels[1:-1]:
        tr = d != held
        m = DoseResponseModel(n_boot=1).fit(dose_um[tr], pathways[tr])
        truth = pathways[d == held].mean(0)
        pred, _ = m.predict(np.array([2**held - 1]))
        nearest = m.means_[np.argmin(np.abs(m.doses_ - held))]
        linear = np.array(
            [np.interp(held, m.doses_, m.means_[:, k]) for k in range(m.means_.shape[1])]
        )
        rows.append(
            {
                "held_out_dose_um": float(2**held - 1),
                "model_r": float(np.corrcoef(pred[0], truth)[0, 1]),
                "model_mae": float(np.abs(pred[0] - truth).mean()),
                "nearest_dose_mae": float(np.abs(nearest - truth).mean()),
                "linear_interp_mae": float(np.abs(linear - truth).mean()),
            }
        )
    return pd.DataFrame(rows)
