"""Shared test fixtures.

The ``FakeSpectronClient`` stands in for a real Spectron deployment so the tests
run offline. It matches the async surface of ``SpectronClient`` and keeps every
stored item in a list, scoped by the fields passed on each call.
"""

from __future__ import annotations

from typing import Any

import pytest

from spectron_openai_agents.config import MemoryScope


def _keywords(text: str) -> set[str]:
    """Split text into lowercased words of three or more letters.

    This gives the fake client a crude form of overlap-based recall, enough to
    exercise the integration without a real semantic index.
    """
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {word for word in cleaned.split() if len(word) >= 3}


class FakeSpectronClient:
    """In-memory stand-in for ``SpectronClient`` used across the tests."""

    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []
        self.forgotten: list[dict[str, Any]] = []

    async def remember(
        self,
        content: str,
        scope: MemoryScope | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        memory_type: str | None = None,
    ) -> str:
        self.stored.append(
            {
                "content": content,
                "scope": scope,
                "metadata": metadata,
                "memory_type": memory_type,
            }
        )
        return "Stored in memory."

    async def recall(
        self,
        query: str,
        scope: MemoryScope | None = None,
        *,
        limit: int = 5,
    ) -> str:
        terms = _keywords(query)
        matches = [
            item["content"]
            for item in self.stored
            if item["scope"] == scope and terms & _keywords(item["content"])
        ]
        return "\n".join(matches[:limit])

    async def context(self, query: str, scope: MemoryScope | None = None) -> str:
        recalled = await self.recall(query, scope, limit=100)
        return f"Context for '{query}':\n{recalled}" if recalled else ""

    async def reflect(
        self, scope: MemoryScope | None = None, *, focus: str | None = None
    ) -> str:
        items = [item["content"] for item in self.stored if item["scope"] == scope]
        return "; ".join(items)

    async def forget(self, target: str, scope: MemoryScope | None = None) -> str:
        self.forgotten.append({"target": target, "scope": scope})
        return "Removed from memory."


@pytest.fixture
def fake_client() -> FakeSpectronClient:
    return FakeSpectronClient()
