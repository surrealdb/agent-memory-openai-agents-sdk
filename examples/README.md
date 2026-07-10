# Examples

Each script is runnable on its own. They need a working Spectron endpoint and an
OpenAI API key set in the environment:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export SPECTRON_URL="https://your-spectron-endpoint"
export SPECTRON_NAMESPACE="your-namespace"
export SPECTRON_DATABASE="your-database"
export SPECTRON_TOKEN="your-token"   # optional for local instances
```

Install the package with the Spectron SDK before running them:

```bash
pip install -e "..[spectron]"
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
