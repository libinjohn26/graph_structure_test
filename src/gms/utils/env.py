"""Minimal .env loading, so endpoints and keys stay out of the repo.

Deliberately not python-dotenv: the need is a handful of KEY=value lines, and
a dependency that reads credentials is one more thing to audit. Existing
environment variables always win, so a shell export overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Load KEY=value pairs from `path` (default: `.env` at the repo root).

    Returns the names that were set, never the values — callers log the result
    and a secret should not end up in a terminal scrollback or CI log.
    """
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return []

    loaded = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
