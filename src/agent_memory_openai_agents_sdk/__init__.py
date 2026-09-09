"""AgentMemory memory for the OpenAI Agents SDK.

This package connects agents built with the OpenAI Agents SDK to AgentMemory,
SurrealDB's memory and knowledge layer. It offers two ways to use memory:

- Function tools the agent calls itself. See :func:`get_agent_memory_tools`.
- Automatic memory around a run. See :func:`run_with_memory` and
  :class:`AgentMemoryMemoryHooks`.

Both talk to AgentMemory through :class:`AgentMemoryClient`, scoped by
:class:`MemoryScope`.
"""

from __future__ import annotations

from .client import AgentMemoryClient
from .config import MemoryScope, AgentMemorySettings
from .hooks import AgentMemoryMemoryHooks, run_with_memory
from .memory_instructions import memory_instructions
from .tools import DEFAULT_OPERATIONS, get_agent_memory_tools

__version__ = "0.1.0"

__all__ = [
    "AgentMemoryClient",
    "MemoryScope",
    "AgentMemorySettings",
    "get_agent_memory_tools",
    "DEFAULT_OPERATIONS",
    "AgentMemoryMemoryHooks",
    "run_with_memory",
    "memory_instructions",
    "__version__",
]
