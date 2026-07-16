"""Spectron memory for the OpenAI Agents SDK.

This package connects agents built with the OpenAI Agents SDK to Spectron,
SurrealDB's memory and knowledge layer. It offers two ways to use memory:

- Function tools the agent calls itself. See :func:`get_spectron_tools`.
- Automatic memory around a run. See :func:`run_with_memory` and
  :class:`SpectronMemoryHooks`.

Both talk to Spectron through :class:`SpectronClient`, scoped by
:class:`MemoryScope`.
"""

from __future__ import annotations

from .client import SpectronClient
from .config import MemoryScope, SpectronSettings
from .hooks import SpectronMemoryHooks, run_with_memory
from .memory_instructions import memory_instructions
from .tools import DEFAULT_OPERATIONS, get_spectron_tools

__version__ = "0.1.0"

__all__ = [
    "SpectronClient",
    "MemoryScope",
    "SpectronSettings",
    "get_spectron_tools",
    "DEFAULT_OPERATIONS",
    "SpectronMemoryHooks",
    "run_with_memory",
    "memory_instructions",
    "__version__",
]
