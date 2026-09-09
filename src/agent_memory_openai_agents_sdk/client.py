"""Adapter around the official AgentMemory SDK (``surrealdb`` 3.x).

This is the only module in the package that imports the AgentMemory SDK. Every
other module (tools, hooks, instructions) talks to AgentMemory through the
``AgentMemoryClient`` surface defined here, so a change in the SDK only affects
this file.

The SDK ships two clients, ``Memory`` (blocking) and ``AsyncMemory``. Both
expose the same operations. This adapter builds the async client by default and
awaits its methods, presenting a uniform async surface. The five operations map
onto SDK methods as follows:

- ``remember`` -> ``remember``
- ``recall``   -> ``recall``
- ``context``  -> ``query_context``
- ``reflect``  -> ``reflect``
- ``forget``   -> ``forget``

Scope is applied per call: ``session_id`` maps to the SDK ``session_id``,
``user_id`` maps to ``on_behalf_of``, and ``agent_id`` maps to ``scopes`` on
writes and ``lens`` on reads.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from .config import MemoryScope, AgentMemorySettings


async def _resolve(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it as is.

    Lets ``AgentMemoryClient`` wrap either ``AsyncMemory`` (async methods) or
    ``Memory`` (blocking methods) without changing its own async surface.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _stringify(result: Any) -> str:
    """Normalize an arbitrary value into a string for a model."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from an object attribute or a dict key."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _render_hits(hits: Any) -> str:
    """Render a recall response's hits into newline-separated text."""
    if not hits:
        return ""
    lines: list[str] = []
    for hit in hits:
        text = None
        for key in ("text", "content", "preview", "snippet", "summary"):
            text = _field(hit, key)
            if text:
                break
        lines.append(str(text) if text else _stringify(hit))
    return "\n".join(line for line in lines if line)


class AgentMemoryClient:
    """Async wrapper around a AgentMemory SDK client.

    Construct it from environment variables, from explicit settings, or from an
    SDK client you already hold:

        client = AgentMemoryClient.from_env()
        client = AgentMemoryClient.from_settings(settings)
        client = AgentMemoryClient.from_sdk(existing_sdk_client)
    """

    def __init__(self, sdk_client: Any) -> None:
        """Wrap an already-constructed AgentMemory SDK client.

        Prefer the ``from_*`` constructors unless you have a reason to pass the
        SDK client directly.
        """
        self._sdk = sdk_client

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_sdk(cls, sdk_client: Any) -> "AgentMemoryClient":
        """Wrap an SDK client (``AsyncMemory`` or ``Memory``) directly."""
        return cls(sdk_client)

    @classmethod
    def from_settings(cls, settings: AgentMemorySettings) -> "AgentMemoryClient":
        """Build a client from explicit connection settings."""
        return cls(_build_sdk_client(settings))

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "AgentMemoryClient":
        """Build a client from ``AGENT_MEMORY_*`` environment variables."""
        return cls.from_settings(AgentMemorySettings.from_env(environ))

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    async def remember(
        self,
        content: str,
        scope: MemoryScope | None = None,
        *,
        memory_category: str | None = None,
        labels: list[str] | None = None,
    ) -> str:
        """Write ``content`` into AgentMemory memory.

        Args:
            content: The text to store.
            scope: Memory partition to write to.
            memory_category: Optional AgentMemory memory category, for example
                ``"semantic"``, ``"episodic"``, or ``"preference"``.
            labels: Optional labels to attach to the stored memory.

        Returns:
            A short confirmation string describing what was stored.
        """
        kwargs = self._write_scope(scope)
        if memory_category is not None:
            kwargs["memory_category"] = memory_category
        if labels is not None:
            kwargs["labels"] = labels
        result = await _resolve(self._sdk.remember(content, **kwargs))
        return _stringify(_field(result, "preview")) or "Stored in memory."

    async def recall(
        self,
        query: str,
        scope: MemoryScope | None = None,
        *,
        limit: int | None = None,
    ) -> str:
        """Search memory for content relevant to ``query``.

        Args:
            query: What to look for.
            scope: Memory partition to search.
            limit: Maximum number of memories to return (the SDK ``k``).

        Returns:
            The matching memories rendered as text, or an empty string when
            nothing matches.
        """
        kwargs = self._read_scope(scope)
        if limit is not None:
            kwargs["k"] = limit
        result = await _resolve(self._sdk.recall(query, **kwargs))
        return _render_hits(_field(result, "hits"))

    async def context(
        self,
        query: str,
        scope: MemoryScope | None = None,
    ) -> str:
        """Assemble a context block for ``query`` from stored memory.

        Backed by the SDK ``query_context`` method. Where ``recall`` returns
        individual matches, this returns a single block AgentMemory has already
        ranked and stitched together for use in a prompt.
        """
        kwargs = self._read_scope(scope, include_session=False)
        result = await _resolve(self._sdk.query_context(query, **kwargs))
        return _stringify(_field(result, "context"))

    async def reflect(
        self,
        query: str,
        scope: MemoryScope | None = None,
        *,
        persist: bool = False,
    ) -> str:
        """Run a synthesis pass over stored memory relevant to ``query``.

        Args:
            query: The topic to reflect on.
            scope: Memory partition to reflect over.
            persist: Store the resulting attributes back into memory.

        Returns:
            The synthesized reflection as text.
        """
        kwargs = self._principal_scope(scope)
        result = await _resolve(self._sdk.reflect(query, persist=persist, **kwargs))
        return _stringify(_field(result, "reflection"))

    async def forget(
        self,
        query: str,
        scope: MemoryScope | None = None,
        *,
        purge: bool = False,
    ) -> str:
        """Remove memory matching ``query``.

        Args:
            query: A description of what to remove.
            scope: Memory partition to remove from.
            purge: Permanently purge rather than soft-delete.

        Returns:
            A short confirmation string.
        """
        kwargs = self._principal_scope(scope)
        result = await _resolve(self._sdk.forget(query, purge=purge, **kwargs))
        deleted = _field(result, "deleted")
        if isinstance(deleted, (list, tuple, set)):
            return f"Removed {len(deleted)} item(s) from memory."
        if isinstance(deleted, int):
            return f"Removed {deleted} item(s) from memory."
        return "Removed from memory."

    async def close(self) -> None:
        """Close the underlying SDK client if it supports it."""
        close = getattr(self._sdk, "close", None)
        if close is not None:
            await _resolve(close())

    # ------------------------------------------------------------------
    # Scope mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _principal_scope(scope: MemoryScope | None) -> dict[str, Any]:
        """Scope arguments accepted by every operation."""
        kwargs: dict[str, Any] = {}
        if scope and scope.user_id:
            kwargs["on_behalf_of"] = scope.user_id
        return kwargs

    def _write_scope(self, scope: MemoryScope | None) -> dict[str, Any]:
        """Scope arguments for writes (remember): session and scopes."""
        kwargs = self._principal_scope(scope)
        if scope and scope.session_id:
            kwargs["session_id"] = scope.session_id
        if scope and scope.agent_id:
            kwargs["scopes"] = scope.agent_id
        return kwargs

    def _read_scope(
        self, scope: MemoryScope | None, *, include_session: bool = True
    ) -> dict[str, Any]:
        """Scope arguments for reads (recall, context): lens and session."""
        kwargs = self._principal_scope(scope)
        if include_session and scope and scope.session_id:
            kwargs["session_id"] = scope.session_id
        if scope and scope.agent_id:
            kwargs["lens"] = scope.agent_id
        return kwargs


def _build_sdk_client(settings: AgentMemorySettings) -> Any:
    """Construct the underlying AgentMemory SDK client from settings.

    Uses ``AsyncMemory`` so the adapter can await its methods. The client is
    created but does not open a connection until an operation runs.
    """
    try:
        from surrealdb.memory import AsyncMemory
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "The AgentMemory SDK is required to build a client from settings. "
            "Install it with `pip install surrealdb`, or pass an "
            "already-constructed SDK client to AgentMemoryClient.from_sdk()."
        ) from exc

    return AsyncMemory(
        settings.context,
        endpoint=settings.endpoint,
        api_key=settings.api_key,
    )
