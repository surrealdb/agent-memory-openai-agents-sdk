"""Inject Spectron memory into an agent's system prompt.

The OpenAI Agents SDK lets ``Agent(instructions=...)`` be a callable that builds
the system prompt at run time. :func:`memory_instructions` returns such a
callable, so an agent can carry a summary of what it remembers about the current
scope without the caller wiring anything up per turn.

The callable does not have access to the run input, so it injects a general
memory summary rather than a query-targeted one. For recall aimed at a specific
message, use :func:`spectron_openai_agents.run_with_memory` instead.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .client import SpectronClient
from .config import MemoryScope

MEMORY_HEADER = "# What you remember"

InstructionsFn = Callable[[Any, Any], Awaitable[str]]


def memory_instructions(
    base_instructions: str,
    client: SpectronClient,
    scope: MemoryScope | None = None,
    *,
    focus: str | None = None,
) -> InstructionsFn:
    """Build a dynamic ``instructions`` callable that prepends stored memory.

    The returned callable recalls a memory summary for ``scope`` and appends it
    to ``base_instructions`` under a "What you remember" heading. When there is
    nothing to recall, the base instructions are returned unchanged.

    Args:
        base_instructions: The agent's normal system prompt.
        client: The Spectron client to read memory from.
        scope: Memory partition to summarize.
        focus: Optional topic. When set, a context block for that topic is
            injected. When unset, a reflection over all memory in scope is used.

    Returns:
        An async callable suitable for ``Agent(instructions=...)``.
    """
    resolved_scope = scope or MemoryScope()

    async def _instructions(run_context: Any, agent: Any) -> str:
        if focus is not None:
            memory = await client.context(focus, resolved_scope)
        else:
            memory = await client.reflect(resolved_scope)
        if not memory:
            return base_instructions
        return f"{base_instructions}\n\n{MEMORY_HEADER}\n{memory}"

    return _instructions
