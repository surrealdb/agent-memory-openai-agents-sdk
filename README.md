# Agent Memory for the OpenAI Agents SDK

Give agents built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
a durable, shared memory backed by [Agent Memory](https://surrealdb.com/agent-memory),
SurrealDB's memory and knowledge layer.

Agent Memory stores facts, past turns, preferences, and knowledge as a graph with temporal
awareness, and lets several agents read and write the same memory. This package wires that
into the Agents SDK so an agent can remember across runs and recall what it needs before it
answers.

## How it works

There are two ways to use memory, and they compose:

1. **Function tools the agent calls itself.** The agent decides when to `remember`, `recall`,
   `context`, `reflect`, or `forget`. This is the pattern used by the mem0 and cognee
   integrations.
2. **Automatic memory around a run.** `run_with_memory` recalls memory relevant to the input,
   injects it into the prompt, runs the agent, and stores the result. The agent needs no memory
   tools of its own.

Both talk to Agent Memory through a single `AgentMemoryClient`, scoped by a `MemoryScope`
(`agent_id`, `session_id`, `user_id`).

## Installation

```bash
pip install agent-memory-openai-agents-sdk
```

This pulls in the Agent Memory SDK (`surrealdb`), which the integration
uses to reach a deployment.

Set the connection and model credentials:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export AGENT_MEMORY_ENDPOINT="https://your-agent_memory-endpoint"
export AGENT_MEMORY_CONTEXT="your-memory-context"
export AGENT_MEMORY_API_KEY="your-api-key"   # optional for local, unsecured instances
```

## Usage

### Memory as function tools

```python
from agents import Agent, Runner
from agent_memory_openai_agents_sdk import get_agent_memory_tools

agent = Agent(
    name="assistant",
    instructions=(
        "You are a helpful assistant. Use recall to check memory before you "
        "answer, and use remember to store anything worth keeping."
    ),
    tools=get_agent_memory_tools(session_id="user-123"),
)

Runner.run_sync(agent, "My name is Ada and I work on databases.")

result = Runner.run_sync(agent, "What do you know about me?")
print(result.final_output)
```

`get_agent_memory_tools` builds the tools from `AGENT_MEMORY_*` environment variables by default. Pass a
`client=` to use an explicit `AgentMemoryClient`, and `include=` to choose a subset of operations.

### Automatic memory around a run

```python
import asyncio
from agents import Agent
from agent_memory_openai_agents_sdk import MemoryScope, run_with_memory

agent = Agent(name="assistant", instructions="You are a helpful assistant.")
scope = MemoryScope(session_id="user-123")

async def main():
    await run_with_memory(agent, "My name is Ada.", scope=scope)

    result = await run_with_memory(agent, "What is my name?", scope=scope)
    print(result.final_output)

asyncio.run(main())
```

### Persisting output with hooks

To save an agent's output while running it yourself, attach `AgentMemoryHooks`:

```python
from agents import Runner
from agent_memory_openai_agents_sdk import AgentMemoryClient, AgentMemoryHooks, MemoryScope

client = AgentMemoryClient.from_env()
hooks = AgentMemoryHooks(client, MemoryScope(session_id="user-123"))

await Runner.run(agent, "Summarize our project decisions.", hooks=hooks)
```

## Operations

| Operation  | Purpose                                                        |
| ---------- | -------------------------------------------------------------- |
| `remember` | Store a fact, preference, or event.                            |
| `recall`   | Search memory for content relevant to a query.                 |
| `context`  | Assemble a single ranked context block for a query.            |
| `reflect`  | Synthesize stored memory into a higher-level summary.          |
| `forget`   | Remove matching memory.                                        |

## Multi-agent shared memory

Agents that share a `MemoryScope` read and write the same Agent Memory, so knowledge one agent
stores is available to another. See
[`examples/04_multi_agent_shared_memory.py`](examples/04_multi_agent_shared_memory.py).

## Examples

Runnable scripts live in [`examples/`](examples/):

- `01_quickstart.py` stores a fact and recalls it with the memory tools.
- `02_function_tools.py` exercises the full tool set.
- `03_automatic_memory.py` uses `run_with_memory` with no memory tools on the agent.
- `04_multi_agent_shared_memory.py` shares one memory across two agents.

## Configuration notes

This integration targets the Agent Memory client in `surrealdb` 3.x, which exposes `AgentMemory` and
`AsyncMemory`. The SDK is imported lazily, only when a client is built from settings or the
environment, and all SDK calls live in `src/agent_memory_openai_agents_sdk/client.py`. If a future SDK
release changes a method name or constructor argument, that file is the single place to adjust; the
tools, hooks, and examples do not change.

The five operations map onto SDK methods as `remember`, `recall`, `context` (the SDK's
`query_context`), `reflect`, and `forget`. Scope is applied per call: `session_id` maps to the SDK
`session_id`, `user_id` to `on_behalf_of`, and `agent_id` to `scopes` on writes and `lens` on reads.

## License

Licensed under the [Apache License 2.0](LICENSE).
