"""Two agents sharing one Spectron memory through a handoff.

A research agent gathers facts and stores them. It then hands off to a writer
agent that reads the same memory to produce a summary. Because both agents use
the same MemoryScope, knowledge stored by one is available to the other.

Required environment variables:
    OPENAI_API_KEY
    SPECTRON_URL, SPECTRON_NAMESPACE, SPECTRON_DATABASE
    SPECTRON_TOKEN (optional)

Run it with:
    python examples/04_multi_agent_shared_memory.py
"""

from agents import Agent, Runner

from spectron_openai_agents import get_spectron_tools

# Both agents share this scope, so they read and write the same memory.
SHARED_SESSION = "shared-project"


def main() -> None:
    writer = Agent(
        name="writer",
        instructions=(
            "You write short project briefs. Recall everything in memory about "
            "the project and turn it into a two sentence summary."
        ),
        tools=get_spectron_tools(session_id=SHARED_SESSION, include=("recall", "context")),
    )

    researcher = Agent(
        name="researcher",
        instructions=(
            "You collect facts about the project and store each one with "
            "remember. When the user asks for a written summary, hand off to "
            "the writer."
        ),
        tools=get_spectron_tools(session_id=SHARED_SESSION, include=("remember", "recall")),
        handoffs=[writer],
    )

    # The researcher stores facts.
    Runner.run_sync(
        researcher,
        "Note these facts: the project is called Orbit, it ships in Q3, and "
        "Grace leads the design.",
    )

    # Asking for a summary triggers a handoff to the writer, which reads the
    # same memory the researcher wrote.
    result = Runner.run_sync(researcher, "Write me the project brief.")
    print(result.final_output)


if __name__ == "__main__":
    main()
