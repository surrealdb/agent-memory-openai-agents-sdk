# Spectron for the OpenAI Agents SDK

Give agents built with the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
a durable, shared memory backed by [Spectron](https://surrealdb.com/platform/spectron),
SurrealDB's memory and knowledge layer.

Spectron stores facts, past turns, preferences, and knowledge as a graph with temporal
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

Both talk to Spectron through a single `SpectronClient`, scoped by a `MemoryScope`
(`agent_id`, `session_id`, `user_id`).

## Installation

```bash
pip install spectron-openai-agents
```

This pulls in the Spectron SDK (`surrealdb[spectron]`), which the integration
uses to reach a deployment.

Set the connection and model credentials:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export SPECTRON_URL="https://your-spectron-endpoint"
export SPECTRON_NAMESPACE="your-namespace"
export SPECTRON_DATABASE="your-database"
export SPECTRON_TOKEN="your-token"   # optional for local, unsecured instances
```

## Usage

### Memory as function tools

```python
from agents import Agent, Runner
from spectron_openai_agents import get_spectron_tools

agent = Agent(
    name="assistant",
    instructions=(
        "You are a helpful assistant. Use recall to check memory before you "
        "answer, and use remember to store anything worth keeping."
    ),
    tools=get_spectron_tools(session_id="user-123"),
)

Runner.run_sync(agent, "My name is Ada and I work on databases.")

result = Runner.run_sync(agent, "What do you know about me?")
print(result.final_output)
```

`get_spectron_tools` builds the tools from `SPECTRON_*` environment variables by default. Pass a
`client=` to use an explicit `SpectronClient`, and `include=` to choose a subset of operations.

### Automatic memory around a run

```python
import asyncio
from agents import Agent
from spectron_openai_agents import MemoryScope, run_with_memory

agent = Agent(name="assistant", instructions="You are a helpful assistant.")
scope = MemoryScope(session_id="user-123")

async def main():
    await run_with_memory(agent, "My name is Ada.", scope=scope)

    result = await run_with_memory(agent, "What is my name?", scope=scope)
    print(result.final_output)

asyncio.run(main())
```

### Persisting output with hooks

To save an agent's output while running it yourself, attach `SpectronMemoryHooks`:

```python
from agents import Runner
from spectron_openai_agents import SpectronClient, SpectronMemoryHooks, MemoryScope

client = SpectronClient.from_env()
hooks = SpectronMemoryHooks(client, MemoryScope(session_id="user-123"))

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

Agents that share a `MemoryScope` read and write the same Spectron memory, so knowledge one agent
stores is available to another. See
[`examples/04_multi_agent_shared_memory.py`](examples/04_multi_agent_shared_memory.py).

## Examples

Runnable scripts live in [`examples/`](examples/):

- `01_quickstart.py` stores a fact and recalls it with the memory tools.
- `02_function_tools.py` exercises the full tool set.
- `03_automatic_memory.py` uses `run_with_memory` with no memory tools on the agent.
- `04_multi_agent_shared_memory.py` shares one memory across two agents.

## Configuration notes

The Spectron SDK is imported lazily, only when a client is built from settings or the
environment. If the released SDK exposes different client method names or a different constructor,
`src/spectron_openai_agents/client.py` is the single place to adjust; the tools, hooks, and
examples do not change.

## License

Licensed under the [Apache License 2.0](LICENSE).
