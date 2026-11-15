"""Config loader and path helpers for the Open OKF Enterprise RAG project."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Read config.yaml once; env vars (KEY.upper()) override YAML values."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    for key in list(cfg.keys()):
        env_key = key.upper()
        if env_key in os.environ:
            env_val = os.environ[env_key]
            original = cfg[key]
            if isinstance(original, bool):
                cfg[key] = env_val.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(original, int) and not isinstance(original, bool):
                try:
                    cfg[key] = int(env_val)
                except ValueError:
                    cfg[key] = env_val
            elif isinstance(original, float):
                try:
                    cfg[key] = float(env_val)
                except ValueError:
                    cfg[key] = env_val
            else:
                cfg[key] = env_val
    return cfg


def okf_bundle_dir() -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg.get("okf_bundle_dir", "okf-bundle")


def source_pdfs_dir() -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg.get("source_pdfs_dir", "source_pdfs")


def concepts_dir() -> Path:
    return okf_bundle_dir() / "concepts"


def images_dir() -> Path:
    return okf_bundle_dir() / "assets" / "images"


def tables_dir() -> Path:
    return okf_bundle_dir() / "assets" / "tables"


def ensure_parent(path: Path) -> Path:
    """Create parent directories for *path* if missing; return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def assets_registry_dir() -> Path:
    """Canonical store for deduplicated image binaries."""
    path = okf_bundle_dir() / "assets" / "registry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    path = PROJECT_ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
