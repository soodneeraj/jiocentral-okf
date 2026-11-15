"""Elasticsearch Q&A cache — store answers for 1 hour to skip Milvus/LLM on repeats."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from app.logger import get_logger
from app.utils import load_config

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_es_client = None
_index_ready = False


def _normalize_query(query: str) -> str:
    return _WHITESPACE_RE.sub(" ", (query or "").strip().lower())


def cache_key(query: str, history: list[dict[str, str]] | None = None) -> str:
    """Stable SHA-256 key from normalized query (+ prior user turn if any)."""
    parts = [_normalize_query(query)]
    if history:
        for msg in reversed(history):
            if msg.get("role") == "user":
                parts.insert(0, _normalize_query(str(msg.get("content") or "")))
                break
    raw = "||".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_es_client():
    """Lazy singleton Elasticsearch client. Returns None if cache disabled/unavailable."""
    global _es_client
    cfg = load_config()
    if not cfg.get("elasticsearch_cache_enabled", True):
        return None
    if _es_client is not None:
        return _es_client
    try:
        from elasticsearch import Elasticsearch

        url = cfg.get("elasticsearch_url") or "https://localhost:9200"
        user = cfg.get("elasticsearch_user") or ""
        password = cfg.get("elasticsearch_password") or ""
        verify = bool(cfg.get("elasticsearch_verify_certs", False))
        kwargs: dict[str, Any] = {
            "hosts": [url],
            "verify_certs": verify,
            "ssl_show_warn": False,
            "request_timeout": 10,
        }
        if user:
            kwargs["basic_auth"] = (user, password)
        _es_client = Elasticsearch(**kwargs)
        # Probe once
        if not _es_client.ping():
            logger.warning("Elasticsearch ping failed at %s", url)
            _es_client = None
            return None
        return _es_client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Elasticsearch client init failed: %s", exc)
        _es_client = None
        return None


def _ensure_index(client) -> bool:
    global _index_ready
    if _index_ready:
        return True
    cfg = load_config()
    index = cfg.get("elasticsearch_cache_index") or "okf_qa_cache"
    try:
        if not client.indices.exists(index=index):
            client.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "cache_key": {"type": "keyword"},
                        "query": {"type": "text"},
                        "query_normalized": {"type": "keyword"},
                        "answer": {"type": "text", "index": False},
                        "sources": {"type": "object", "enabled": False},
                        "images": {"type": "object", "enabled": False},
                        "created_at": {"type": "date"},
                        "expires_at": {"type": "date"},
                    }
                },
            )
            logger.info("Created Elasticsearch cache index %s", index)
        _index_ready = True
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to ensure ES cache index: %s", exc)
        return False


def get_cached_answer(
    query: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Return cached RAG payload if present and not expired; else None."""
    cfg = load_config()
    if not cfg.get("elasticsearch_cache_enabled", True):
        return None
    client = get_es_client()
    if client is None or not _ensure_index(client):
        return None

    index = cfg.get("elasticsearch_cache_index") or "okf_qa_cache"
    key = cache_key(query, history)
    now = datetime.now(timezone.utc).isoformat()
    try:
        resp = client.search(
            index=index,
            size=1,
            query={
                "bool": {
                    "must": [
                        {"term": {"cache_key": key}},
                        {"range": {"expires_at": {"gt": now}}},
                    ]
                }
            },
        )
        hits = (resp.get("hits") or {}).get("hits") or []
        if not hits:
            return None
        src = hits[0].get("_source") or {}
        payload = {
            "answer": src.get("answer") or "",
            "sources": src.get("sources") or [],
            "images": src.get("images") or [],
            "cache": "hit",
            "cache_expires_at": src.get("expires_at"),
        }
        logger.info("ES cache HIT for key=%s…", key[:12])
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("ES cache lookup failed: %s", exc)
        return None


def put_cached_answer(
    query: str,
    result: dict[str, Any],
    history: list[dict[str, str]] | None = None,
) -> None:
    """Store a RAG answer in Elasticsearch with TTL (default 1 hour)."""
    cfg = load_config()
    if not cfg.get("elasticsearch_cache_enabled", True):
        return
    client = get_es_client()
    if client is None or not _ensure_index(client):
        return

    index = cfg.get("elasticsearch_cache_index") or "okf_qa_cache"
    ttl = int(cfg.get("elasticsearch_cache_ttl_seconds") or 3600)
    key = cache_key(query, history)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=ttl)
    doc = {
        "cache_key": key,
        "query": query,
        "query_normalized": _normalize_query(query),
        "answer": result.get("answer") or "",
        "sources": result.get("sources") or [],
        "images": result.get("images") or [],
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
    }
    try:
        client.index(index=index, id=key, document=doc, refresh="wait_for")
        logger.info("ES cache STORE key=%s… ttl=%ds", key[:12], ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ES cache store failed: %s", exc)


def elasticsearch_health() -> dict[str, Any]:
    """Probe ES for /health — never raises."""
    cfg = load_config()
    if not cfg.get("elasticsearch_cache_enabled", True):
        return {"status": "disabled"}
    try:
        client = get_es_client()
        if client is None:
            return {"status": "error: unreachable or misconfigured"}
        info = client.info()
        index = cfg.get("elasticsearch_cache_index") or "okf_qa_cache"
        exists = client.indices.exists(index=index)
        return {
            "status": "ok",
            "cluster_name": info.get("cluster_name"),
            "version": (info.get("version") or {}).get("number"),
            "cache_index": index,
            "cache_index_exists": bool(exists),
            "ttl_seconds": int(cfg.get("elasticsearch_cache_ttl_seconds") or 3600),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": f"error: {exc}"}
