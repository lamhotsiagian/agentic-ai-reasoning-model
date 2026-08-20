"""Model access for the reasoning engine.

Two models by design (Chapter 7): a reasoner and a structurally different,
smaller critic. A verifier that shares the generator's failure modes adds
cost and no signal.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, ValidationError

from app.config import settings

T = TypeVar("T", bound=BaseModel)

_MODEL_CACHE: dict[tuple[str, bool], BaseChatModel] = {}


def create_model(model_name: str, *, streaming: bool = False,
                 temperature: float = 0.0) -> BaseChatModel:
    """Build (and memoise) a chat model handle."""
    key = (f"{model_name}:{temperature}", streaming)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if settings.model_provider == "ollama":
        from langchain_ollama import ChatOllama

        model: BaseChatModel = ChatOllama(
            model=model_name,
            base_url=settings.model_base_url,
            temperature=temperature,
            streaming=streaming,
        )
    else:
        from langchain.chat_models import init_chat_model

        model = init_chat_model(
            model=model_name,
            model_provider=settings.model_provider,
            api_key=settings.api_key or "local",
            base_url=settings.model_base_url or None,
            temperature=temperature,
            streaming=streaming,
        )

    _MODEL_CACHE[key] = model
    return model


def reasoner(temperature: float = 0.0, streaming: bool = False) -> BaseChatModel:
    return create_model(settings.reasoner_model, temperature=temperature,
                        streaming=streaming)


def critic(temperature: float = 0.0) -> BaseChatModel:
    """Deliberately a different, smaller model, because judging is easier than
    generating, and different weights mean less correlated errors."""
    return create_model(settings.critic_model, temperature=temperature)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


@dataclass
class Structured:
    """Result envelope. `ok=False` carries the validation error verbatim so
    the caller can feed it back for repair, which beats resampling."""

    ok: bool
    value: Any = None
    error: str | None = None
    raw: str = ""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def _extract_json(text: str) -> str:
    """Small models fence their JSON or prepend prose. Recover both."""
    if (m := _FENCE.search(text)) is not None:
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _usage_from(message: Any, elapsed_ms: int) -> Usage:
    meta = getattr(message, "usage_metadata", None) or {}
    return Usage(
        prompt_tokens=int(meta.get("input_tokens", 0)),
        completion_tokens=int(meta.get("output_tokens", 0)),
        latency_ms=elapsed_ms,
    )


async def structured_call(prompt: str, schema: type[T], *,
                          system: str | None = None,
                          model: BaseChatModel | None = None,
                          temperature: float = 0.0
                          ) -> tuple[Structured, Usage]:
    """Call a model and validate the response against a Pydantic schema.

    Returns the validation error rather than raising: the orchestrator
    decides whether to repair, resample, or degrade.
    """
    model = model or reasoner(temperature=temperature)
    messages: list[Any] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(
        content=f"{prompt}\n\nRespond with JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema(), indent=2)}"
    ))

    started = time.perf_counter()
    response = await model.ainvoke(messages)
    elapsed = int((time.perf_counter() - started) * 1000)
    usage = _usage_from(response, elapsed)
    raw = str(response.content)

    try:
        payload = json.loads(_extract_json(raw))
        return Structured(ok=True, value=schema(**payload), raw=raw), usage
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.debug(f"structured_call validation failed: {exc}")
        return Structured(ok=False, error=str(exc)[:500], raw=raw), usage


async def text_call(prompt: str, *, system: str | None = None,
                    model: BaseChatModel | None = None,
                    temperature: float = 0.0) -> tuple[str, Usage]:
    model = model or reasoner(temperature=temperature)
    messages: list[Any] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    started = time.perf_counter()
    response = await model.ainvoke(messages)
    elapsed = int((time.perf_counter() - started) * 1000)
    return str(response.content), _usage_from(response, elapsed)
