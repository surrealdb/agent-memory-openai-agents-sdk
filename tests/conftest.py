"""Shared test fixtures.

The ``FakeAgentMemoryClient`` stands in for a real Agent Memory deployment so the tests
run offline. It matches the async surface of ``AgentMemoryClient`` and keeps every
stored item in a list, scoped by the fields passed on each call.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_memory_openai_agents_sdk.config import MemoryScope


def _keywords(text: str) -> set[str]:
    """Split text into lowercased words of three or more letters.

    This gives the fake client a crude form of overlap-based recall, enough to
    exercise the integration without a real semantic index.
    """
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {word for word in cleaned.split() if len(word) >= 3}


class FakeAgentMemoryClient:
    """In-memory stand-in for ``AgentMemoryClient`` used across the tests."""

    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []
        self.forgotten: list[dict[str, Any]] = []

    async def remember(
        self,
        content: str,
        scope: MemoryScope | None = None,
        *,
        memory_category: str | None = None,
        labels: list[str] | None = None,
    ) -> str:
        self.stored.append(
            {
                "content": content,
                "scope": scope,
                "memory_category": memory_category,
                "labels": labels,
            }
        )
        return "Stored in memory."

    async def recall(
        self,
        query: str,
        scope: MemoryScope | None = None,
        *,
        limit: int | None = None,
    ) -> str:
        terms = _keywords(query)
        matches = [
            item["content"]
            for item in self.stored
            if item["scope"] == scope and terms & _keywords(item["content"])
        ]
        return "\n".join(matches[: limit or len(matches)])

    async def context(self, query: str, scope: MemoryScope | None = None) -> str:
        recalled = await self.recall(query, scope)
        return f"Context for '{query}':\n{recalled}" if recalled else ""

    async def reflect(
        self, query: str, scope: MemoryScope | None = None, *, persist: bool = False
    ) -> str:
        items = [item["content"] for item in self.stored if item["scope"] == scope]
        return "; ".join(items)

    async def forget(
        self, query: str, scope: MemoryScope | None = None, *, purge: bool = False
    ) -> str:
        self.forgotten.append({"query": query, "scope": scope, "purge": purge})
        return "Removed from memory."


@pytest.fixture
def fake_client() -> FakeAgentMemoryClient:
    return FakeAgentMemoryClient()
