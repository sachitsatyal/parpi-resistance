"""Command-line interface: ``parpi <command> --config configs/default.yaml``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

from parpi.config import load_config

CONFIG = click.option(
    "--config",
    "-c",
    default="configs/default.yaml",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)


@click.group()
@click.option("-v", "--verbose", is_flag=True)
def main(verbose: bool) -> None:
    """Reproducible single-cell PARP-inhibitor resistance models."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@main.command()
@click.argument("accession")
def inspect(accession: str) -> None:
    """List supplementary files of a GEO series (to fill in a dataset config)."""
    from parpi.data.download import list_geo_files

    for name in list_geo_files(accession):
        click.echo(name)


@main.command()
@CONFIG
def prepare(config: str) -> None:
    """Download, QC and featurize all datasets."""
    from parpi.pipeline import prepare as run

    run(load_config(config))


@main.command()
@CONFIG
def train(config: str) -> None:
    """Cross-validate all models, fit them on all data, select the release model."""
    from parpi.pipeline import train as run

    run(load_config(config))


@main.command()
@CONFIG
def evaluate(config: str) -> None:
    """Score trained models on the external datasets."""
    from parpi.pipeline import evaluate_external

    click.echo(json.dumps(evaluate_external(load_config(config)), indent=2))


@main.command()
@CONFIG
@click.option("--hf-repo", default="OWNER/parpi-resistance", show_default=True)
def report(config: str, hf_repo: str) -> None:
    """Write the model card for the selected model from the run's metrics."""
    from parpi.report import write_model_card

    click.echo(str(write_model_card(load_config(config), hf_repo)))


@main.command()
@CONFIG
@click.option("--min-spearman", default=0.8, show_default=True)
@click.option("--max-mae-ratio", default=0.8, show_default=True)
@click.option("--min-external-auroc", default=0.65, show_default=True)
@click.option("--min-line-auroc", default=0.7, show_default=True)
def gate(
    config: str,
    min_spearman: float,
    max_mae_ratio: float,
    min_external_auroc: float,
    min_line_auroc: float,
) -> None:
    """Fail (exit 1) unless the selected model passes the release quality gate."""
    from parpi.pipeline import quality_gate

    failures = quality_gate(
        load_config(config), min_spearman, max_mae_ratio, min_external_auroc, min_line_auroc
    )
    for f in failures:
        click.echo(f"GATE FAILED: {f}", err=True)
    if failures:
        raise SystemExit(1)
    click.echo("quality gate passed")


@main.command()
@CONFIG
@click.pass_context
def reproduce(ctx: click.Context, config: str) -> None:
    """Run every stage: prepare -> train -> evaluate -> report."""
    for cmd in (prepare, train, evaluate, report):
        ctx.invoke(cmd, config=config)


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default="outputs/models/hierarchical",
    show_default=True,
    help="Bundle folder or hf://org/repo[@revision]",
)
@click.option("--out", "-o", default="predictions.csv", show_default=True)
@click.option("--no-qc", is_flag=True, help="Skip QC filtering (input already filtered).")
def predict(input_path: str, model: str, out: str, no_qc: bool) -> None:
    """Score cells in an .h5ad / 10x / CSV file."""
    from parpi.bundle import ModelBundle
    from parpi.data.readers import read_any

    bundle = ModelBundle.load(model)
    df = bundle.predict(read_any(input_path), apply_qc=not no_qc)
    df.to_csv(out, index=False)
    click.echo(f"{len(df)} cells scored, gene coverage {df.attrs['gene_coverage']:.0%} -> {out}")


@main.command("dose-curve")
@click.option("--model", "-m", required=True)
@click.option("--doses", default="0,1,2.5,5,10,20,40,80,160,320", show_default=True)
@click.option("--out", "-o", default="dose_curve.csv", show_default=True)
def dose_curve(model: str, doses: str, out: str) -> None:
    """Predicted pathway activity across olaparib concentrations (µM)."""
    from parpi.bundle import ModelBundle

    d = np.array([float(v) for v in doses.split(",")])
    ModelBundle.load(model).dose_curve(d).to_csv(out, index=False)
    click.echo(out)


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--repo", required=True, help="Hugging Face model repo id, e.g. org/name")
@click.option("--revision-tag", default=None, help="Tag the Hub commit (e.g. v0.1.0)")
def push(model: str, repo: str, revision_tag: str | None) -> None:
    """Upload a bundle to the Hugging Face Hub (needs HF_TOKEN)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    info = api.upload_folder(
        folder_path=model,
        repo_id=repo,
        commit_message=f"Upload model bundle from {Path(model).name}",
    )
    if revision_tag:
        api.create_tag(repo, tag=revision_tag, revision=info.oid, exist_ok=True)
    click.echo(f"pushed {model} -> https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
