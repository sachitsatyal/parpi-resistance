"""Gradio web app (Hugging Face Space and `docker run`)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
from logic import dose_curve_figure, load_bundle, score_file, score_histogram

DESCRIPTION = """
# PARP-inhibitor resistance score (single-cell)
Upload single-cell counts (**.h5ad**, 10x **.h5**, or a **CSV** of cells × genes with HGNC
symbols). Each cell gets a score for how far it sits along the olaparib resistance continuum,
an uncertainty, and the pathways driving it. **Research use only — not a diagnostic.**
"""


def run(file, apply_qc):  # type: ignore[no-untyped-def]
    if file is None:
        raise gr.Error("Upload a file first.")
    try:
        pred, summary, status = score_file(file, load_bundle(), apply_qc=apply_qc)
    except ValueError as err:
        raise gr.Error(str(err)) from err
    out = Path(tempfile.mkdtemp()) / "parpi_predictions.csv"
    pred.to_csv(out, index=False)
    return status, summary, score_histogram(pred), pred.head(200), str(out)


def curves(pathways):  # type: ignore[no-untyped-def]
    if not pathways:
        raise gr.Error("Pick at least one pathway.")
    return dose_curve_figure(load_bundle(), pathways[:6])


def build() -> gr.Blocks:
    bundle = load_bundle()
    top = bundle.pathway_importance()["pathway"].head(40).tolist()
    with gr.Blocks(title="PARPi resistance") as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tab("Score cells"):
            with gr.Row():
                f = gr.File(label="Counts file", file_types=[".h5ad", ".h5", ".csv", ".gz"])
                qc = gr.Checkbox(value=True, label="Apply the model's QC filters")
            btn = gr.Button("Score", variant="primary")
            status = gr.Markdown()
            summary = gr.Dataframe(label="Per-sample summary")
            hist = gr.Plot(label="Score distribution")
            table = gr.Dataframe(label="First 200 cells")
            dl = gr.File(label="Download all predictions")
            btn.click(run, [f, qc], [status, summary, hist, table, dl])
        with gr.Tab("Dose response"):
            gr.Markdown(
                "Pathway activity versus olaparib dose, with 90% bootstrap bands. "
                "The grey region is outside the measured doses (not validated)."
            )
            pw = gr.Dropdown(top, value=top[:3], multiselect=True, label="Pathways (max 6)")
            plot = gr.Plot()
            gr.Button("Plot").click(curves, pw, plot)
        with gr.Tab("Model card"):
            card = Path(os.environ.get("PARPI_CARD", "")) if os.environ.get("PARPI_CARD") else None
            gr.Markdown(
                card.read_text()
                if card and card.exists()
                else "See the model repository on the Hugging Face Hub for the full card."
            )
    return demo


if __name__ == "__main__":
    build().queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860"))
    )
