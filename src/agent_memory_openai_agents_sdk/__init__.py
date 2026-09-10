"""Agent Memory for the OpenAI Agents SDK.

This package connects agents built with the OpenAI Agents SDK to Agent Memory,
SurrealDB's memory and knowledge layer. It offers two ways to use memory:

- Function tools the agent calls itself. See :func:`get_agent_memory_tools`.
- Automatic memory around a run. See :func:`run_with_memory` and
  :class:`AgentMemoryHooks`.

Both talk to Agent Memory through :class:`AgentMemoryClient`, scoped by
:class:`MemoryScope`.
"""

from __future__ import annotations

from .client import AgentMemoryClient
from .config import MemoryScope, AgentMemorySettings
from .hooks import AgentMemoryHooks, run_with_memory
from .memory_instructions import memory_instructions
from .tools import DEFAULT_OPERATIONS, get_agent_memory_tools

__version__ = "0.2.0"

__all__ = [
    "AgentMemoryClient",
    "MemoryScope",
    "AgentMemorySettings",
    "get_agent_memory_tools",
    "DEFAULT_OPERATIONS",
    "AgentMemoryHooks",
    "run_with_memory",
    "memory_instructions",
    "__version__",
]
