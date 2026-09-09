# Examples

Each script is runnable on its own. They need a working AgentMemory endpoint and an
OpenAI API key set in the environment:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export AGENT_MEMORY_ENDPOINT="https://your-agent_memory-endpoint"
export AGENT_MEMORY_CONTEXT="your-memory-context"
export AGENT_MEMORY_API_KEY="your-api-key"   # optional for local instances
```

Install the package before running them:

```bash
pip install -e ..
```

| Script                             | What it shows                                                    |
| ---------------------------------- | ---------------------------------------------------------------- |
| `01_quickstart.py`                 | Memory tools; store a fact in one run, recall it in the next.    |
| `02_function_tools.py`             | The full tool set: remember, recall, context, reflect, forget.   |
| `03_automatic_memory.py`           | `run_with_memory` with no memory tools on the agent.             |
| `04_multi_agent_shared_memory.py`  | Two agents sharing one memory scope through a handoff.           |

Run one with:

```bash
python examples/01_quickstart.py
```
