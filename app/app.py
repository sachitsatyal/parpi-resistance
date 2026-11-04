"""Gradio web app (Hugging Face Space and `docker run`)."""

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path

import gradio as gr
from logic import (
    ALL,
    dose_curve_figure,
    load_bundle,
    page_view,
    sample_choices,
    score_file,
    score_histogram,
)

EXAMPLE = Path(__file__).parent / "examples" / "kuramochi_10x_example.h5ad"

DESCRIPTION = """
# PARP-inhibitor adaptation score (single-cell, in vitro)
Upload single-cell counts (**.h5ad**, 10x **.h5**, or a **CSV** of cells × genes with HGNC
symbols), or press **Try an example**. Each cell gets a score for how far it has adapted to
olaparib (0 = untreated, 1 = top training dose) and the Hallmark pathways behind it.
Validated on ovarian cancer cell lines; **not** a tumor or patient resistance score.
Compare samples within one experiment. **Research use only.**
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
    view, caption, _pages = page_view(pred)
    return (
        status,
        summary,
        score_histogram(pred),
        view,
        str(out),
        pred,
        gr.Dropdown(choices=sample_choices(pred), value=ALL),
        1,
        caption,
    )


def show_page(pred, sample, page, step=0):  # type: ignore[no-untyped-def]
    """Page `page + step` of the chosen sample, clamped to the pages that exist."""
    if pred is None:
        raise gr.Error("Score a file first.")
    _, _, pages = page_view(pred, sample, 1)
    target = int(min(max(int(page or 1) + step, 1), pages))
    view, caption, _ = page_view(pred, sample, target)
    return view, target, caption


def run_example(apply_qc):  # type: ignore[no-untyped-def]
    return run(str(EXAMPLE), apply_qc)


def card_text(bundle) -> str:  # type: ignore[no-untyped-def]
    """The model card stored with the bundle, without its YAML header."""
    card = bundle.folder / "README.md" if bundle.folder else None
    if not card or not card.exists():
        return "See the model repository on the Hugging Face Hub for the full card."
    text = card.read_text(encoding="utf-8")
    return text.split("---", 2)[2] if text.startswith("---") else text


def curves(pathways):  # type: ignore[no-untyped-def]
    if not pathways:
        raise gr.Error("Pick at least one pathway.")
    return dose_curve_figure(load_bundle(), pathways[:6])


# Blue accents instead of Gradio's default orange. Gradio 6 moved `theme` from Blocks()
# to launch(); pass it wherever this installed version accepts it.
THEME = gr.themes.Default(primary_hue="blue", secondary_hue="slate")
_THEME_IN_BLOCKS = "theme" in inspect.signature(gr.Blocks.__init__).parameters


def build() -> gr.Blocks:
    bundle = load_bundle()
    top = bundle.pathway_importance()["pathway"].head(40).tolist()
    blocks_kw = {"theme": THEME} if _THEME_IN_BLOCKS else {}
    with gr.Blocks(title="PARPi adaptation score", **blocks_kw) as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Tab("Score cells"):
            with gr.Row():
                f = gr.File(label="Counts file", file_types=[".h5ad", ".h5", ".csv", ".gz"])
                qc = gr.Checkbox(value=True, label="Apply the model's QC filters")
            with gr.Row():
                btn = gr.Button("Score", variant="primary")
                ex_btn = gr.Button(
                    "Try an example: Kuramochi on 10x, untreated / 10 µM / 320 µM (unseen data)",
                    visible=EXAMPLE.exists(),
                )
            status = gr.Markdown()
            summary = gr.Dataframe(label="Per-sample summary")
            hist = gr.Plot(label="Score distribution")
            preds = gr.State()
            with gr.Row():
                sample_dd = gr.Dropdown([ALL], value=ALL, label="Show cells from", scale=3)
                prev_btn = gr.Button("◀ Previous", scale=1)
                page = gr.Number(value=1, precision=0, label="Page", scale=1)
                next_btn = gr.Button("Next ▶", scale=1)
            caption = gr.Markdown()
            table = gr.Dataframe(label="Cells (100 per page)")
            dl = gr.File(label="Download all predictions")
            outputs = [status, summary, hist, table, dl, preds, sample_dd, page, caption]
            btn.click(run, [f, qc], outputs)
            ex_btn.click(run_example, [qc], outputs)
            nav = [table, page, caption]
            sample_dd.input(lambda p, s: show_page(p, s, 1), [preds, sample_dd], nav)
            prev_btn.click(lambda p, s, n: show_page(p, s, n, -1), [preds, sample_dd, page], nav)
            next_btn.click(lambda p, s, n: show_page(p, s, n, +1), [preds, sample_dd, page], nav)
            page.submit(show_page, [preds, sample_dd, page], nav)
        with gr.Tab("Dose response"):
            gr.Markdown(
                "Pathway activity versus olaparib dose, with 90% bootstrap bands. "
                "The grey region is outside the measured doses (not validated)."
            )
            pw = gr.Dropdown(top, value=top[:3], multiselect=True, label="Pathways (max 6)")
            plot = gr.Plot()
            gr.Button("Plot").click(curves, pw, plot)
        with gr.Tab("Model card"):
            gr.Markdown(card_text(bundle))
    return demo


if __name__ == "__main__":
    launch_kw = {} if _THEME_IN_BLOCKS else {"theme": THEME}
    build().queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), **launch_kw
    )
