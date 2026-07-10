"""Transparent memory with run_with_memory and no memory tools.

The agent here has no memory tools. run_with_memory recalls memory relevant to
each input, injects it into the prompt, runs the agent, and stores the input and
output afterward. Memory works without the agent knowing anything about it.

Required environment variables:
    OPENAI_API_KEY
    SPECTRON_URL, SPECTRON_NAMESPACE, SPECTRON_DATABASE
    SPECTRON_TOKEN (optional)

Run it with:
    python examples/03_automatic_memory.py
"""

import asyncio

from agents import Agent

from spectron_openai_agents import MemoryScope, run_with_memory


async def main() -> None:
    agent = Agent(
        name="assistant",
        instructions="You are a concise, helpful assistant.",
    )
    scope = MemoryScope(session_id="auto-memory-user")

    # First turn: state a preference. It is stored automatically.
    await run_with_memory(agent, "I prefer answers in metric units.", scope=scope)

    # Second turn: the preference is recalled and injected before the agent runs.
    result = await run_with_memory(
        agent, "How far is it from London to Paris?", scope=scope
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
