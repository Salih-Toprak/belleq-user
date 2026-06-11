"""Uygulama ömrü boyunca paylaşılan tekil nesneler."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_startup_time: float = 0.0
_state_store: Any = None
_global_store: Any = None
_vectordb: Any = None
_embedder: Any = None
_lifecycle_retriever: Any = None
_pipeline: Any = None
_conversation_store: Any = None
_conversation_capture: Any = None


def get_state_store() -> Any:
    return _state_store


def get_global_store() -> Any:
    return _global_store


def get_vectordb() -> Any:
    return _vectordb


def get_embedder() -> Any:
    return _embedder


def get_lifecycle_retriever() -> Any:
    return _lifecycle_retriever


def get_pipeline() -> Any:
    return _pipeline


def get_conversation_store() -> Any:
    return _conversation_store


def get_conversation_capture() -> Any:
    return _conversation_capture


def get_uptime() -> float:
    if _startup_time <= 0.0:
        return 0.0
    return time.monotonic() - _startup_time
