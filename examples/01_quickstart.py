"""Quickstart: give an agent memory tools and watch it recall across runs.

The agent is created with the Agent Memory tools. The first run stores a
fact. The second run, with no reminder in the prompt, recalls it from memory.

Required environment variables:
    OPENAI_API_KEY
    AGENT_MEMORY_ENDPOINT, AGENT_MEMORY_CONTEXT
    AGENT_MEMORY_API_KEY (optional)

Run it with:
    python examples/01_quickstart.py
"""

from agents import Agent, Runner

from agent_memory_openai_agents_sdk import get_agent_memory_tools


def main() -> None:
    agent = Agent(
        name="assistant",
        instructions=(
            "You are a helpful assistant with a long-term memory. "
            "Use recall to check what you already know before answering, and "
            "use remember to store facts and preferences the user shares."
        ),
        tools=get_agent_memory_tools(session_id="quickstart-user"),
    )

    # First run: the user shares a fact worth remembering.
    Runner.run_sync(agent, "My name is Ada and I work on databases.")

    # Second run: nothing in the prompt repeats the fact, so the agent has to
    # recall it from Agent Memory.
    result = Runner.run_sync(agent, "What do you know about me?")
    print(result.final_output)


if __name__ == "__main__":
    main()
