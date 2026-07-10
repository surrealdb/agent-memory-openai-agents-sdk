"""Configuration objects for the Spectron OpenAI Agents integration.

This module holds two small, dependency-free pieces of state:

- ``MemoryScope`` identifies which slice of memory an operation reads from or
  writes to. Spectron partitions memory by agent, session, and user, so the
  same scope is threaded through every call the integration makes.
- ``SpectronSettings`` collects the connection details needed to reach a
  Spectron deployment, with a helper to load them from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemoryScope:
    """Identifies the slice of Spectron memory an operation applies to.

    All fields are optional. A scope with no fields set targets the default
    memory partition for the connected namespace and database. Setting
    ``session_id`` keeps a single conversation isolated, while a shared
    ``agent_id`` or ``user_id`` lets several sessions or agents read and write
    the same memory.

    Attributes:
        agent_id: Identifier for the agent that owns or shares the memory.
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
        url: Base URL of the Spectron endpoint, for example
            ``https://cloud.surrealdb.com`` or ``http://localhost:8000``.
        namespace: SurrealDB namespace that holds the memory tables.
        database: SurrealDB database that holds the memory tables.
        token: Bearer token or API key used to authenticate. Optional for
            local development against an unsecured instance.
    """

    url: str
    namespace: str
    database: str
    token: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "SpectronSettings":
        """Build settings from environment variables.

        Reads ``SPECTRON_URL``, ``SPECTRON_NAMESPACE``, ``SPECTRON_DATABASE``,
        and the optional ``SPECTRON_TOKEN``. The OpenAI Agents SDK reads
        ``OPENAI_API_KEY`` on its own, so it is not handled here.

        Args:
            environ: Mapping to read from. Defaults to ``os.environ``.

        Returns:
            A populated ``SpectronSettings`` instance.

        Raises:
            ValueError: If any required variable is missing.
        """
        env = os.environ if environ is None else environ
        missing = [
            name
            for name in ("SPECTRON_URL", "SPECTRON_NAMESPACE", "SPECTRON_DATABASE")
            if not env.get(name)
        ]
        if missing:
            raise ValueError(
                "Missing required Spectron environment variables: "
                + ", ".join(missing)
                + ". Set them or pass a SpectronClient explicitly."
            )
        return cls(
            url=env["SPECTRON_URL"],
            namespace=env["SPECTRON_NAMESPACE"],
            database=env["SPECTRON_DATABASE"],
            token=env.get("SPECTRON_TOKEN"),
        )
