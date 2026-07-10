"""Adapter around the official Spectron SDK (``surrealdb[spectron]``).

This is the only module in the package that imports the Spectron SDK. Every
other module (tools, hooks, instructions) talks to Spectron through the
``SpectronClient`` surface defined here. If the released SDK exposes different
method names or constructor arguments, this file is the single place to update
and the rest of the package is unaffected.

The five operations mirror Spectron's own vocabulary:

- ``remember`` writes content into memory.
- ``recall`` searches memory for content relevant to a query.
- ``context`` assembles a ready-to-use context block for a query.
- ``reflect`` runs a synthesis pass over stored memory.
- ``forget`` removes matching memory.

All methods are async and return plain strings so their output can be handed
straight back to a language model.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from .config import MemoryScope, SpectronSettings


async def _resolve(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it as is.

    The Spectron SDK may expose sync or async methods depending on version and
    transport. Resolving here lets ``SpectronClient`` present a uniform async
    surface regardless.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def _stringify(result: Any) -> str:
    """Normalize an arbitrary SDK return value into a string for a model."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


class SpectronClient:
    """Async wrapper around a Spectron SDK client.

    Construct it from environment variables, from explicit settings, or from an
    SDK client you already hold:

        client = SpectronClient.from_env()
        client = SpectronClient.from_settings(settings)
        client = SpectronClient.from_sdk(existing_sdk_client)
    """

    def __init__(self, sdk_client: Any) -> None:
        """Wrap an already-constructed Spectron SDK client.

        Prefer the ``from_*`` constructors unless you have a reason to pass the
        SDK client directly.
        """
        self._sdk = sdk_client

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_sdk(cls, sdk_client: Any) -> "SpectronClient":
        """Wrap an SDK client that the caller has already configured."""
        return cls(sdk_client)

    @classmethod
    def from_settings(cls, settings: SpectronSettings) -> "SpectronClient":
        """Build a client from explicit connection settings."""
        return cls(_build_sdk_client(settings))

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "SpectronClient":
        """Build a client from ``SPECTRON_*`` environment variables."""
        return cls.from_settings(SpectronSettings.from_env(environ))

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------
    async def remember(
        self,
        content: str,
        scope: MemoryScope | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> str:
        """Write ``content`` into Spectron memory.

        Args:
            content: The text to store.
            scope: Memory partition to write to.
            metadata: Optional structured metadata to attach.
            memory_type: Optional Spectron memory type, for example
                ``"semantic"``, ``"episodic"``, or ``"preference"``.

        Returns:
            A short confirmation string describing what was stored.
        """
        kwargs: dict[str, Any] = self._scope_kwargs(scope)
        if metadata is not None:
            kwargs["metadata"] = metadata
        if memory_type is not None:
            kwargs["memory_type"] = memory_type
        result = await _resolve(self._sdk.remember(content, **kwargs))
        return _stringify(result) or "Stored in memory."

    async def recall(
        self,
        query: str,
        scope: MemoryScope | None = None,
        *,
        limit: int = 5,
    ) -> str:
        """Search memory for content relevant to ``query``.

        Args:
            query: What to look for.
            scope: Memory partition to search.
            limit: Maximum number of memories to return.

        Returns:
            The matching memories rendered as text, or an empty string when
            nothing matches.
        """
        kwargs = self._scope_kwargs(scope)
        kwargs["limit"] = limit
        result = await _resolve(self._sdk.recall(query, **kwargs))
        return _stringify(result)

    async def context(
        self,
        query: str,
        scope: MemoryScope | None = None,
    ) -> str:
        """Assemble a context block for ``query`` from stored memory.

        Where ``recall`` returns individual matches, ``context`` returns a
        single block Spectron has already ranked and stitched together for use
        in a prompt.
        """
        kwargs = self._scope_kwargs(scope)
        result = await _resolve(self._sdk.context(query, **kwargs))
        return _stringify(result)

    async def reflect(
        self,
        scope: MemoryScope | None = None,
        *,
        focus: str | None = None,
    ) -> str:
        """Run a synthesis pass over stored memory.

        Args:
            scope: Memory partition to reflect over.
            focus: Optional topic to steer the synthesis toward.

        Returns:
            The synthesized summary as text.
        """
        kwargs = self._scope_kwargs(scope)
        if focus is not None:
            kwargs["focus"] = focus
        result = await _resolve(self._sdk.reflect(**kwargs))
        return _stringify(result)

    async def forget(
        self,
        target: str,
        scope: MemoryScope | None = None,
    ) -> str:
        """Remove memory matching ``target``.

        Args:
            target: A description or identifier of what to remove.
            scope: Memory partition to remove from.

        Returns:
            A short confirmation string.
        """
        kwargs = self._scope_kwargs(scope)
        result = await _resolve(self._sdk.forget(target, **kwargs))
        return _stringify(result) or "Removed from memory."

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _scope_kwargs(scope: MemoryScope | None) -> dict[str, Any]:
        """Turn a scope into keyword arguments for an SDK call."""
        if scope is None:
            return {}
        return dict(scope.as_dict())


def _build_sdk_client(settings: SpectronSettings) -> Any:
    """Construct the underlying Spectron SDK client from settings.

    This is the second half of the isolation boundary. It assumes the Spectron
    extra exposes a ``Spectron`` client importable from the ``surrealdb``
    package that accepts a URL, namespace, database, and optional token. Adjust
    this function to match the released SDK if the entry point differs.
    """
    try:
        from surrealdb import Spectron  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "The Spectron SDK is required to build a client from settings. "
            "Install it with `pip install surrealdb[spectron]`, or pass an "
            "already-constructed SDK client to SpectronClient.from_sdk()."
        ) from exc

    return Spectron(
        settings.url,
        namespace=settings.namespace,
        database=settings.database,
        token=settings.token,
    )
