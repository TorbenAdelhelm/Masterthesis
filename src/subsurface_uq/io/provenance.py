from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import subprocess
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local file without loading it into memory."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: str | Path) -> str | None:
    """Return ``git rev-parse HEAD`` for ``path`` when it belongs to a Git checkout."""

    root = Path(path).expanduser().resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def runtime_versions(packages: Iterable[str] = ("numpy", "torch", "scipy", "noise")) -> dict[str, object]:
    """Record platform/interpreter and selected package versions without local paths."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }
