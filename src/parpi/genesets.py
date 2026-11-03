"""Gene-set (pathway) libraries.

Sources:
  * ``enrichr:<Library>`` — fetched once from Enrichr (e.g. ``MSigDB_Hallmark_2020``,
    ``Reactome_2022``), cached as a GMT file and pinned by SHA-256.
  * a path to a local ``.gmt`` file.

The *same* pathway definitions are used for every dataset, so pathway features
are comparable across doses, cell lines and cancer types. (The original project
re-ran enrichment per dose, giving each dose graph a different node set.)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import scipy.sparse as sp

ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={lib}"


@dataclass(frozen=True)
class GeneSets:
    names: tuple[str, ...]
    members: tuple[tuple[str, ...], ...]
    source: str
    sha256: str

    def __len__(self) -> int:
        return len(self.names)

    def filter(self, universe: np.ndarray, min_size: int, max_size: int) -> GeneSets:
        """Keep sets with between min_size and max_size genes present in ``universe``."""
        uni = set(universe.tolist())
        keep = [
            (n, tuple(g for g in m if g in uni))
            for n, m in zip(self.names, self.members, strict=True)
        ]
        keep = [(n, m) for n, m in keep if min_size <= len(m) <= max_size]
        if not keep:
            raise ValueError(
                "no gene sets survive filtering against the data's genes; check that gene "
                "identifiers are HGNC symbols"
            )
        return GeneSets(
            tuple(n for n, _ in keep), tuple(m for _, m in keep), self.source, self.sha256
        )

    def genes(self) -> np.ndarray:
        return np.array(sorted({g for m in self.members for g in m}))

    def membership(self, genes: np.ndarray) -> sp.csr_matrix:
        """Binary genes x sets matrix aligned to ``genes``."""
        index = {g: i for i, g in enumerate(genes)}
        rows, cols = [], []
        for j, m in enumerate(self.members):
            for g in m:
                if g in index:
                    rows.append(index[g])
                    cols.append(j)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(genes), len(self))
        )

    def to_gmt(self, path: Path) -> None:
        lines = ["\t".join([n, "", *m]) for n, m in zip(self.names, self.members, strict=True)]
        path.write_text("\n".join(lines) + "\n")


def parse_gmt(text: str, source: str) -> GeneSets:
    names, members = [], []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 3 or not parts[0]:
            continue
        genes = tuple(dict.fromkeys(p.split(",")[0] for p in parts[2:] if p))
        names.append(parts[0])
        members.append(genes)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return GeneSets(tuple(names), tuple(members), source, digest)


def load_gene_sets(spec: str, cache_dir: Path) -> GeneSets:
    """Resolve one or more ``+``-joined specs into a single library."""
    parts = [s.strip() for s in spec.split("+") if s.strip()]
    libs = [_load_one(p, cache_dir) for p in parts]
    if len(libs) == 1:
        return libs[0]
    names = tuple(n for lib in libs for n in lib.names)
    members = tuple(m for lib in libs for m in lib.members)
    digest = hashlib.sha256("".join(lib.sha256 for lib in libs).encode()).hexdigest()
    return GeneSets(names, members, spec, digest)


def _load_one(spec: str, cache_dir: Path) -> GeneSets:
    if spec.startswith("enrichr:"):
        lib = spec.split(":", 1)[1]
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{lib}.gmt"
        lock = cache_dir / "genesets.lock.json"
        pins: dict[str, str] = json.loads(lock.read_text()) if lock.exists() else {}
        if not path.exists():
            resp = requests.get(ENRICHR_URL.format(lib=lib), timeout=120)
            resp.raise_for_status()
            path.write_text(resp.text)
        gs = parse_gmt(path.read_text(), spec)
        if lib in pins and pins[lib] != gs.sha256:
            raise RuntimeError(f"gene-set library {lib} changed (sha256 mismatch with lock file)")
        pins[lib] = gs.sha256
        lock.write_text(json.dumps(pins, indent=2, sort_keys=True))
        return gs
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"gene-set file not found: {spec}")
    return parse_gmt(path.read_text(), str(path))
