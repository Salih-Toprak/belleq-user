"""Route bağımlılıkları ve kimlik doğrulama."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Header, HTTPException, Request

import app.config as app_config
from app.query.pipeline import QueryPipeline

logger = logging.getLogger(__name__)


async def require_master(x_master_key: str = Header(default="", alias="X-Master-Key" )) -> None:
    s = app_config.settings
    if s.master_api_key and x_master_key != s.master_api_key:
        raise HTTPException(status_code=403, detail="Invalid master key")


async def require_user(x_api_key: str = Header(default="")) -> None:
    s = app_config.settings
    if s.user_api_key and x_api_key != s.user_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")


def get_pipeline(request: Request) -> QueryPipeline:
    return request.app.state.pipeline


def get_global_store(request: Request) -> Any:
    return request.app.state.global_store


def get_state_store(request: Request) -> Any:
    return request.app.state.state_store


def get_vectordb(request: Request) -> Any:
    return getattr(request.app.state, "vectordb", None)


def get_conversation_store(request: Request) -> Any:
    return getattr(request.app.state, "conversation_store", None)


def get_conversation_capture(request: Request) -> Any:
    return getattr(request.app.state, "conversation_capture", None)
