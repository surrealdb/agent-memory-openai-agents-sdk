"""Tests for SpectronMemoryHooks and run_with_memory."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from spectron_openai_agents import (
    MemoryScope,
    SpectronMemoryHooks,
    run_with_memory,
)
from spectron_openai_agents.hooks import MEMORY_CONTEXT_HEADER


class FakeRunner:
    """Records the input it is handed and returns a fixed final output."""

    def __init__(self, final_output: str = "the answer") -> None:
        self.final_output = final_output
        self.received_input: Any = None

    async def run(self, agent: Any, input: Any, **kwargs: Any) -> Any:
        self.received_input = input
        return SimpleNamespace(final_output=self.final_output)


async def test_hooks_persist_agent_output(fake_client):
    hooks = SpectronMemoryHooks(fake_client, MemoryScope(session_id="s1"))

    await hooks.on_agent_end(context=None, agent=None, output="Paris is the capital")

    assert len(fake_client.stored) == 1
    assert "Paris is the capital" in fake_client.stored[0]["content"]
    assert fake_client.stored[0]["scope"].session_id == "s1"


async def test_hooks_skip_output_when_disabled(fake_client):
    hooks = SpectronMemoryHooks(fake_client, persist_output=False)

    await hooks.on_agent_end(context=None, agent=None, output="ignored")

    assert fake_client.stored == []


async def test_hooks_persist_tool_results_when_enabled(fake_client):
    hooks = SpectronMemoryHooks(fake_client, persist_tool_results=True)
    tool = SimpleNamespace(name="search")

    await hooks.on_tool_end(context=None, agent=None, tool=tool, result="a result")

    assert len(fake_client.stored) == 1
    assert "Result of search" in fake_client.stored[0]["content"]


async def test_run_with_memory_recalls_injects_and_persists(fake_client):
    scope = MemoryScope(session_id="s1")
    # Seed a memory the recall step should find and inject. It shares the word
    # "theme" with the question below, which the fake client matches on.
    await fake_client.remember("The user's preferred theme is dark mode", scope)

    runner = FakeRunner(final_output="Enabled dark mode")
    result = await run_with_memory(
        agent=SimpleNamespace(name="assistant"),
        input="What theme do I like?",
        client=fake_client,
        scope=scope,
        runner=runner,
    )

    # The recalled memory was injected as a leading system message.
    assert isinstance(runner.received_input, list)
    system_message = runner.received_input[0]
    assert system_message["role"] == "system"
    assert MEMORY_CONTEXT_HEADER in system_message["content"]
    assert "dark mode" in system_message["content"]

    # The final output is surfaced unchanged.
    assert result.final_output == "Enabled dark mode"

    # Both the user input and the output were persisted.
    contents = [item["content"] for item in fake_client.stored]
    assert any("What theme do I like?" in c for c in contents)
    assert any("Enabled dark mode" in c for c in contents)


async def test_run_with_memory_without_matches_does_not_inject(fake_client):
    runner = FakeRunner()
    await run_with_memory(
        agent=SimpleNamespace(name="assistant"),
        input="A brand new question",
        client=fake_client,
        scope=MemoryScope(session_id="s1"),
        runner=runner,
        persist_input=False,
        persist_output=False,
    )

    # Nothing to recall, so the input passes through unchanged.
    assert runner.received_input == "A brand new question"
