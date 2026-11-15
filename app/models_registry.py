"""Configurable local model registry with health checks (Ollama-first)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import ollama

from app.logger import get_logger
from app.utils import load_config

logger = get_logger(__name__)


@dataclass
class ModelSpec:
    role: str
    name: str
    provider: str = "ollama"
    dimension: int | None = None
    endpoint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Abstraction over local inference endpoints (currently Ollama)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        self.ollama_url = str(self.cfg.get("ollama_url") or "http://localhost:11434")

    def specs(self) -> dict[str, ModelSpec]:
        return {
            "chat": ModelSpec(
                role="chat",
                name=str(self.cfg.get("chat_model") or "gemma4:12b"),
                endpoint=self.ollama_url,
                params={
                    "fallback": self.cfg.get("chat_model_fallback"),
                    "num_ctx": self.cfg.get("chat_num_ctx", 4096),
                    "num_predict": self.cfg.get("chat_num_predict", 512),
                },
            ),
            "embedding": ModelSpec(
                role="embedding",
                name=str(self.cfg.get("embedding_model") or "nomic-embed-text:latest"),
                endpoint=self.ollama_url,
                dimension=int(self.cfg.get("embedding_dim") or 768),
            ),
            "vision": ModelSpec(
                role="vision",
                name=str(self.cfg.get("vision_model") or self.cfg.get("chat_model") or ""),
                endpoint=self.ollama_url,
            ),
            "rerank": ModelSpec(
                role="rerank",
                name=str(self.cfg.get("rerank_model") or ""),
                provider=str(self.cfg.get("rerank_provider") or "heuristic"),
                endpoint=self.ollama_url,
            ),
            "ocr": ModelSpec(
                role="ocr",
                name=str(self.cfg.get("ocr_engine") or "paddleocr"),
                provider="local",
            ),
        }

    def client(self) -> ollama.Client:
        return ollama.Client(host=self.ollama_url)

    def list_ollama_models(self) -> list[str]:
        try:
            listing = self.client().list()
            if isinstance(listing, dict):
                models_raw = listing.get("models") or []
            else:
                models_raw = getattr(listing, "models", None) or []
            names: list[str] = []
            for m in models_raw:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or ""
                else:
                    name = getattr(m, "name", None) or getattr(m, "model", "") or ""
                if name:
                    names.append(str(name))
            return names
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama list failed: %s", exc)
            return []

    def health(self) -> dict[str, Any]:
        available = self.list_ollama_models()
        available_l = {n.lower() for n in available}
        specs = self.specs()
        roles: dict[str, Any] = {}
        for role, spec in specs.items():
            if spec.provider == "ollama" and spec.name:
                present = any(
                    spec.name.lower() == n or n.startswith(spec.name.lower().split(":")[0])
                    for n in available_l
                )
                roles[role] = {
                    "name": spec.name,
                    "provider": spec.provider,
                    "available": present or (role == "rerank" and not spec.name),
                    "dimension": spec.dimension,
                }
            else:
                roles[role] = {
                    "name": spec.name,
                    "provider": spec.provider,
                    "available": True,
                    "dimension": spec.dimension,
                }
        return {
            "ollama_url": self.ollama_url,
            "ollama_reachable": bool(available) or self._ping(),
            "models": roles,
            "installed": available,
        }

    def _ping(self) -> bool:
        try:
            self.client().list()
            return True
        except Exception:  # noqa: BLE001
            return False


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
