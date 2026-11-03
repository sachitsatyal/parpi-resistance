"""Deploy the web app to a Hugging Face Space (Docker SDK).

The Space is built from the same Dockerfile and lockfile as the GHCR image, and
pinned to an exact model revision via the PARPI_MODEL variable.

    HF_TOKEN=... python scripts/deploy_space.py --space org/app --model hf://org/model@v0.1.0
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["Dockerfile", "pyproject.toml", "uv.lock", "LICENSE"]
DIRS = ["src", "app"]


def stage(dest: Path) -> None:
    for f in FILES:
        shutil.copy2(ROOT / f, dest / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, dest / d, ignore=shutil.ignore_patterns("__pycache__"))
    # The Space's README carries the Space metadata; the project README is needed by the build.
    shutil.copy2(ROOT / "space" / "README.md", dest / "README.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--model", required=True, help="hf://org/repo@revision")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(args.space, "PARPI_MODEL", args.model)
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.space,
            repo_type="space",
            commit_message=f"Deploy app with model {args.model}",
            delete_patterns=["src/**", "app/**"],  # drop stale code, keep .gitattributes
        )
    print(f"deployed https://huggingface.co/spaces/{args.space} (model {args.model})")


if __name__ == "__main__":
    main()
