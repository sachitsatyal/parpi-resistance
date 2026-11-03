"""Reproducible downloads from GEO.

Every downloaded file is recorded with its SHA-256 in ``MANIFEST.json``. On later
runs the manifest is verified, so a silently changed upstream file fails loudly
instead of quietly changing results.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
import tarfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def geo_suppl_url(accession: str) -> str:
    """GSE206125 -> https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206125/suppl/"""
    if not re.fullmatch(r"GSE\d+", accession):
        raise ValueError(f"not a GEO series accession: {accession}")
    stub = accession[:-3] + "nnn" if len(accession) > 6 else "GSEnnn"
    return f"{GEO_FTP}/{stub}/{accession}/suppl/"


def _session() -> requests.Session:
    """HTTP session that retries timeouts and transient server errors with backoff.

    NCBI's servers are occasionally slow; 5 attempts with 2-32 s waits ride that out.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def list_geo_files(accession: str, timeout: int = 60) -> list[str]:
    resp = _session().get(geo_suppl_url(accession), timeout=timeout)
    resp.raise_for_status()
    names = re.findall(r'href="([^"?/][^"]*)"', resp.text)
    return sorted({n for n in names if not n.startswith(("http", "/"))})


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: int = 600) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _session().get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    tmp.rename(dest)


def safe_extract_tar(archive: Path, dest: Path) -> list[Path]:
    """Extract a tar archive, refusing absolute paths, '..' and links."""
    dest = dest.resolve()
    out: list[Path] = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest)) or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe path in archive: {member.name}")
            if member.isfile():
                tar.extract(member, dest)
                out.append(target)
    return out


def fetch_geo(accession: str, patterns: list[str], dest: Path) -> list[Path]:
    """Download supplementary files of ``accession`` matching ``patterns`` into ``dest``.

    ``*_RAW.tar`` archives are unpacked; patterns are then re-applied to their contents.
    """
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "MANIFEST.json"
    manifest: dict[str, str] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    # Offline path: if every pattern is already satisfied by checksummed local files,
    # skip the network entirely (reruns work without GEO, and checksums still apply).
    cached = {
        p: [n for n in manifest if fnmatch.fnmatch(n, p) and (dest / n).exists()] for p in patterns
    }
    if patterns and all(cached.values()):
        remote: list[str] = []
        wanted = sorted({n for names in cached.values() for n in names})
        log.info("%s: using %d cached file(s), no download needed", accession, len(wanted))
    else:
        remote = list_geo_files(accession)
        wanted = [n for n in remote if any(fnmatch.fnmatch(n, p) for p in patterns)]
    if not wanted:
        raise FileNotFoundError(
            f"no files in {accession} match {patterns}. Available: {remote}. "
            "Run `parpi inspect` and update `files:` in the dataset config."
        )
    paths: list[Path] = []
    for name in wanted:
        local = dest / name
        if not local.exists():
            log.info("downloading %s", name)
            _download(geo_suppl_url(accession) + name, local)
        digest = sha256(local)
        if name in manifest and manifest[name] != digest:
            raise RuntimeError(
                f"checksum mismatch for {name}: manifest {manifest[name]} != file {digest}. "
                "Upstream data changed; delete the file to re-download and review results."
            )
        manifest[name] = digest
        if name.endswith(".tar"):
            members = safe_extract_tar(local, dest / "extracted")
            paths.extend(
                m
                for m in members
                if any(fnmatch.fnmatch(m.name, p) for p in patterns) or name.endswith("_RAW.tar")
            )
        else:
            paths.append(local)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return sorted(paths)
