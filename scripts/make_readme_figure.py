"""Draw the README results figure from a run's metrics (default: the release run).

    uv run python scripts/make_readme_figure.py [outputs/release/metrics]

Writes docs/figures/results.png. Numbers come from cv.json and external.json, never typed.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/release/metrics")
cv = json.loads((folder / "cv.json").read_text())
ext = json.loads((folder / "external.json").read_text())
best = cv["selected_model"]
per = cv["models"][best]["per_dataset"]

GREY, ACCENT, RED, INK, QUIET = "#b8b8b0", "#2a78d6", "#c8553d", "#1a1a1a", "#666666"
labels = {
    "kuramochi_10x": "Kuramochi, 10x platform",
    "pdx_resistance": "PDX tumors (curated)",
    "pdx_resistance_allcells": "PDX tumors (all cells)",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
fig.suptitle(
    "The score transfers to unseen cell lines and platforms, not yet to tumors",
    x=0.02,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)


def fmt(v):
    if v > 0.999 or abs(v - 0.65) < 0.005:  # 0.99988 or 0.6498 must not round to 1.000 / 0.650
        return f"{v:.4f}"
    return f"{v:.3f}" if v < 0.7 else f"{v:.2f}"


def bars(ax, names, vals, cols):
    # Bars start at chance (0.5): their length is how far above chance a test lands.
    ax.barh(names, [v - 0.5 for v in vals], left=0.5, color=cols, height=0.6)
    for i, v in enumerate(vals):
        x = v + 0.008
        if x < 0.66 and x + 0.045 > 0.65:  # keep clear of the dashed 0.65 line
            x = 0.658
        ax.text(x, i, fmt(v), va="center", ha="left", fontsize=10, color=INK, clip_on=False)
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC (0.5 = chance)", fontsize=10, color=QUIET)
    ax.invert_yaxis()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10, colors=INK)


def line_auc(r):
    return next(v for k, v in r.items() if k.startswith("auroc"))


lines = sorted(per, key=lambda k: -line_auc(per[k]))
bars(ax1, lines, [line_auc(per[n]) for n in lines], [ACCENT] * len(lines))
ax1.set_title("Held-out cell line: untreated vs top dose", loc="left", fontsize=11, color=INK)

names = [n for n in labels if n in ext and ext[n].get(best, {}).get("auroc") is not None]
vals = [ext[n][best]["auroc"] for n in names]
cols = [
    ACCENT if ext[n].get("gated") else (RED if v < 0.65 else GREY)
    for n, v in zip(names, vals, strict=True)
]
bars(ax2, [labels[n] for n in names], vals, cols)
ax2.axvline(0.65, color=QUIET, ls="--", lw=1)
ax2.text(0.656, 0.5, "release bar 0.65", fontsize=9, color=QUIET, va="center")
ax2.set_title(
    "External tests: blue = claimed, red/grey = reported only", loc="left", fontsize=11, color=INK
)

fig.tight_layout(rect=(0, 0, 1, 0.92))
out = Path("docs/figures/results.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(out)
