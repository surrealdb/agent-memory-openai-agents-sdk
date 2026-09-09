"""AgentMemory memory operations exposed as OpenAI Agents function tools.

Use :func:`get_agent_memory_tools` to build a list of tools bound to a client and a
memory scope, then pass them to an ``Agent``:

    from agents import Agent
    from agent_memory_openai_agents_sdk import get_agent_memory_tools

    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant with a long-term memory.",
        tools=get_agent_memory_tools(session_id="user-123"),
    )

Each tool is a thin wrapper over a :class:`AgentMemoryClient` method. The client
and scope are captured in a closure so the model only supplies the arguments
that matter to it (content, query, and so on).
"""

from __future__ import annotations

from typing import Any

from agents import function_tool
from agents.tool import FunctionTool

from .client import AgentMemoryClient
from .config import MemoryScope

#: The operations included by default, in the order agents usually reach for them.
DEFAULT_OPERATIONS: tuple[str, ...] = (
    "remember",
    "recall",
    "context",
    "reflect",
    "forget",
)

_default_client: AgentMemoryClient | None = None


def _get_default_client() -> AgentMemoryClient:
    """Return a process-wide client built lazily from the environment."""
    global _default_client
    if _default_client is None:
        _default_client = AgentMemoryClient.from_env()
    return _default_client


def _remember_tool(client: AgentMemoryClient, scope: MemoryScope) -> FunctionTool:
    @function_tool(name_override="remember")
    async def remember(content: str, memory_category: str | None = None) -> str:
        """Store a fact, preference, or event in long-term memory.

        Call this whenever the user shares something worth keeping across turns,
        such as their name, a preference, a decision, or an outcome.

        Args:
            content: The information to store, written as a clear statement.
            memory_category: Optional category, for example "semantic",
                "episodic", or "preference". Leave unset to let AgentMemory decide.
        """
        return await client.remember(content, scope, memory_category=memory_category)

    return remember


def _recall_tool(client: AgentMemoryClient, scope: MemoryScope) -> FunctionTool:
    @function_tool(name_override="recall")
    async def recall(query: str, limit: int = 5) -> str:
        """Search long-term memory for information relevant to a query.

        Call this before answering when the reply may depend on earlier turns,
        stored facts, or user preferences.

        Args:
            query: What to look for in memory.
            limit: Maximum number of memories to return.
        """
        return await client.recall(query, scope, limit=limit)

    return recall


def _context_tool(client: AgentMemoryClient, scope: MemoryScope) -> FunctionTool:
    @function_tool(name_override="context")
    async def context(query: str) -> str:
        """Assemble a ready-to-use context block for a query.

        Prefer this over recall when you want a single, ranked summary of
        everything relevant rather than a list of individual memories.

        Args:
            query: The topic to build context for.
        """
        return await client.context(query, scope)

    return context


def _reflect_tool(client: AgentMemoryClient, scope: MemoryScope) -> FunctionTool:
    @function_tool(name_override="reflect")
    async def reflect(query: str) -> str:
        """Synthesize stored memory into a higher-level summary.

        Use this to consolidate what is known about a topic into a single
        summary, drawing across many stored memories.

        Args:
            query: The topic to reflect on.
        """
        return await client.reflect(query, scope)

    return reflect


def _forget_tool(client: AgentMemoryClient, scope: MemoryScope) -> FunctionTool:
    @function_tool(name_override="forget")
    async def forget(query: str) -> str:
        """Remove information from long-term memory.

        Call this when the user asks to delete or correct something previously
        stored.

        Args:
            query: A description of what to remove.
        """
        return await client.forget(query, scope)

    return forget


_BUILDERS: dict[str, Any] = {
    "remember": _remember_tool,
    "recall": _recall_tool,
    "context": _context_tool,
    "reflect": _reflect_tool,
    "forget": _forget_tool,
}


def get_agent_memory_tools(
    client: AgentMemoryClient | None = None,
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    include: tuple[str, ...] = DEFAULT_OPERATIONS,
) -> list[FunctionTool]:
    """Build AgentMemory memory tools bound to a client and scope.

    Args:
        client: The AgentMemory client to use. Defaults to a client built lazily
            from ``AGENT_MEMORY_*`` environment variables.
        agent_id: Optional agent identifier for the memory scope.
        session_id: Optional session identifier for the memory scope.
        user_id: Optional user identifier for the memory scope.
        include: Which operations to expose, drawn from
            :data:`DEFAULT_OPERATIONS`. Order is preserved in the result.

    Returns:
        A list of function tools ready to pass to an ``Agent``.

    Raises:
        ValueError: If ``include`` names an unknown operation.
    """
    unknown = [name for name in include if name not in _BUILDERS]
    if unknown:
        raise ValueError(
            "Unknown AgentMemory operation(s): "
            + ", ".join(unknown)
            + ". Choose from: "
            + ", ".join(_BUILDERS)
        )

    resolved = client if client is not None else _get_default_client()
    scope = MemoryScope(agent_id=agent_id, session_id=session_id, user_id=user_id)
    return [_BUILDERS[name](resolved, scope) for name in include]
