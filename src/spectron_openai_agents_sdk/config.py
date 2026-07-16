"""Configuration objects for the Spectron OpenAI Agents integration.

This module holds two small, dependency-free pieces of state:

- ``MemoryScope`` identifies which slice of memory an operation reads from or
  writes to. It is threaded through every call the integration makes and mapped
  onto the SDK's scoping arguments in ``client.py``.
- ``SpectronSettings`` collects the connection details needed to reach a
  Spectron deployment, with a helper to load them from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemoryScope:
    """Identifies the slice of Spectron memory an operation applies to.

    All fields are optional. A scope with no fields set targets the whole
    memory context the client is connected to. The fields map onto the SDK's
    scoping arguments in ``client.py``:

    - ``session_id`` maps to the SDK ``session_id`` and keeps a single
      conversation isolated.
    - ``user_id`` maps to ``on_behalf_of``, the principal the memory is for.
    - ``agent_id`` maps to ``scopes`` on writes and ``lens`` on reads, so
      several agents can partition or share a slice of memory.

    Attributes:
        agent_id: Identifier used as the write scope and read lens.
        session_id: Identifier for a single conversation or run.
        user_id: Identifier for the end user the memory belongs to.
    """

    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Return the scope as a dict with only the fields that are set."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    def is_empty(self) -> bool:
        """Return True when no scoping field is set."""
        return not self.as_dict()


@dataclass(frozen=True)
class SpectronSettings:
    """Connection details for a Spectron deployment.

    Attributes:
        endpoint: Base URL of the Spectron endpoint, for example
            ``https://cloud.surrealdb.com`` or ``http://localhost:8000``.
        context: The Spectron memory context to operate in. This is the
            top-level partition the client is bound to.
        api_key: API key used to authenticate. Optional for local development
            against an unsecured instance.
    """

    endpoint: str
    context: str
    api_key: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "SpectronSettings":
        """Build settings from environment variables.

        Reads ``SPECTRON_ENDPOINT`` (``SPECTRON_URL`` is accepted as an alias),
        ``SPECTRON_CONTEXT``, and the optional ``SPECTRON_API_KEY``
        (``SPECTRON_TOKEN`` is accepted as an alias). The OpenAI Agents SDK
        reads ``OPENAI_API_KEY`` on its own, so it is not handled here.

        Args:
            environ: Mapping to read from. Defaults to ``os.environ``.

        Returns:
            A populated ``SpectronSettings`` instance.

        Raises:
            ValueError: If any required variable is missing.
        """
        env = os.environ if environ is None else environ
        endpoint = env.get("SPECTRON_ENDPOINT") or env.get("SPECTRON_URL")
        context = env.get("SPECTRON_CONTEXT")
        api_key = env.get("SPECTRON_API_KEY") or env.get("SPECTRON_TOKEN")

        missing: list[str] = []
        if not endpoint:
            missing.append("SPECTRON_ENDPOINT")
        if not context:
            missing.append("SPECTRON_CONTEXT")
        if missing:
            raise ValueError(
                "Missing required Spectron environment variables: "
                + ", ".join(missing)
                + ". Set them or pass a SpectronClient explicitly."
            )
        return cls(endpoint=endpoint, context=context, api_key=api_key)
