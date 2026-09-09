"""Automatic AgentMemory memory around an agent run.

Two entry points are provided:

- :class:`AgentMemoryMemoryHooks` is a ``RunHooks`` implementation that persists an
  agent's output (and optionally its tool results) as they happen. Attach it to
  any ``Runner.run`` call to save what an agent produces without changing the
  agent itself.
- :func:`run_with_memory` wraps a run end to end: it recalls memory relevant to
  the input, injects it into the prompt, runs the agent, and stores both the
  user input and the final output. This is the transparent path that needs no
  memory tools on the agent.
"""

from __future__ import annotations

from typing import Any

from agents import Runner
from agents.lifecycle import RunHooks

from .client import AgentMemoryClient
from .config import MemoryScope

MEMORY_CONTEXT_HEADER = "Relevant information from memory:"


class AgentMemoryMemoryHooks(RunHooks):
    """Persist agent output to AgentMemory as a run progresses.

    Lifecycle hooks in the Agents SDK are observational, so this class writes to
    memory as a side effect and never alters the prompt. Use it when you run an
    agent through ``Runner`` yourself and want its output saved automatically:

        hooks = AgentMemoryMemoryHooks(client, MemoryScope(session_id="s1"))
        await Runner.run(agent, "Hello", hooks=hooks)
    """

    def __init__(
        self,
        client: AgentMemoryClient,
        scope: MemoryScope | None = None,
        *,
        persist_output: bool = True,
        persist_tool_results: bool = False,
    ) -> None:
        """Configure what the hooks persist.

        Args:
            client: The AgentMemory client to write through.
            scope: Memory partition to write to.
            persist_output: Store each agent's final output when it finishes.
            persist_tool_results: Store the result of every tool call. Off by
                default because AgentMemory memory tools already write on their own.
        """
        self._client = client
        self._scope = scope or MemoryScope()
        self._persist_output = persist_output
        self._persist_tool_results = persist_tool_results

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        """Store the agent's final output as an episodic memory."""
        if not self._persist_output:
            return
        text = _to_text(output)
        if text:
            await self._client.remember(
                f"Assistant response: {text}",
                self._scope,
                memory_category="episodic",
            )

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: Any
    ) -> None:
        """Store a tool result when ``persist_tool_results`` is enabled."""
        if not self._persist_tool_results:
            return
        text = _to_text(result)
        tool_name = getattr(tool, "name", "tool")
        if text:
            await self._client.remember(
                f"Result of {tool_name}: {text}",
                self._scope,
                memory_category="episodic",
            )


async def run_with_memory(
    agent: Any,
    input: str | list[Any],
    *,
    client: AgentMemoryClient | None = None,
    scope: MemoryScope | None = None,
    recall_limit: int = 5,
    use_context: bool = False,
    persist_input: bool = True,
    persist_output: bool = True,
    runner: Any = Runner,
    **run_kwargs: Any,
) -> Any:
    """Run an agent with memory recalled before and persisted after.

    The flow is:

    1. Recall memory relevant to ``input`` (via ``recall`` or, when
       ``use_context`` is set, ``context``).
    2. Prepend the retrieved memory to the input as a system message.
    3. Run the agent through ``runner``.
    4. Store the user input and the final output back into memory.

    Args:
        agent: The agent to run.
        input: The user input, either a string or a list of input items.
        client: The AgentMemory client. Defaults to one built from the environment.
        scope: Memory partition to read from and write to.
        recall_limit: Maximum number of memories to recall.
        use_context: Recall a single ``context`` block instead of a list of
            individual memories.
        persist_input: Store the user input before the run.
        persist_output: Store the final output after the run.
        runner: The runner to use. Defaults to ``Runner``; must expose an async
            ``run`` method with the Agents SDK signature.
        **run_kwargs: Forwarded to ``runner.run`` (for example ``context`` or
            ``max_turns``).

    Returns:
        The ``RunResult`` returned by ``runner.run``.
    """
    resolved_client = client if client is not None else AgentMemoryClient.from_env()
    resolved_scope = scope or MemoryScope()

    query = _input_to_text(input)

    memory = ""
    if query:
        if use_context:
            memory = await resolved_client.context(query, resolved_scope)
        else:
            memory = await resolved_client.recall(
                query, resolved_scope, limit=recall_limit
            )

    if persist_input and query:
        await resolved_client.remember(
            f"User said: {query}",
            resolved_scope,
            memory_category="episodic",
        )

    run_input = _inject_memory(input, memory)
    result = await runner.run(agent, run_input, **run_kwargs)

    if persist_output:
        output_text = _to_text(getattr(result, "final_output", None))
        if output_text:
            await resolved_client.remember(
                f"Assistant response: {output_text}",
                resolved_scope,
                memory_category="episodic",
            )

    return result


def _inject_memory(input: str | list[Any], memory: str) -> str | list[Any]:
    """Prepend a memory block to the input as a system message."""
    if not memory:
        return input

    system_item = {
        "role": "system",
        "content": f"{MEMORY_CONTEXT_HEADER}\n{memory}",
    }

    if isinstance(input, str):
        return [system_item, {"role": "user", "content": input}]
    return [system_item, *input]


def _input_to_text(input: str | list[Any]) -> str:
    """Extract plain text from a string or a list of input items."""
    if isinstance(input, str):
        return input

    parts: list[str] = []
    for item in input:
        content = None
        if isinstance(item, dict):
            content = item.get("content")
        else:
            content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _to_text(value: Any) -> str:
    """Render an arbitrary run or tool output into a trimmed string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
