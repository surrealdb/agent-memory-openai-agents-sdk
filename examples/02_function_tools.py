"""Exercise the full Agent Memory tool set from a single agent.

This gives the agent every memory operation (remember, recall, context,
reflect, forget) and sends a sequence of prompts that lead it to choose the
right one. The agent stores facts, recalls them, synthesizes a summary with
reflect, and removes a fact with forget.

Required environment variables:
    OPENAI_API_KEY
    AGENT_MEMORY_ENDPOINT, AGENT_MEMORY_CONTEXT
    AGENT_MEMORY_API_KEY (optional)

Run it with:
    python examples/02_function_tools.py
"""

from agents import Agent, Runner

from agent_memory_openai_agents_sdk import get_agent_memory_tools


def main() -> None:
    agent = Agent(
        name="memory-assistant",
        instructions=(
            "You manage a personal knowledge base for the user. "
            "Store new facts with remember. Answer questions by recalling or "
            "building context from memory. Use reflect to summarize what you "
            "know, and forget to remove information when asked."
        ),
        tools=get_agent_memory_tools(
            session_id="tools-demo",
            include=("remember", "recall", "context", "reflect", "forget"),
        ),
    )

    prompts = [
        "Remember that our launch date is March 3rd.",
        "Remember that the design review is owned by Grace.",
        "What context do you have about the launch?",
        "Give me a short summary of everything you know so far.",
        "Forget the launch date, it has not been decided yet.",
    ]

    for prompt in prompts:
        result = Runner.run_sync(agent, prompt)
        print(f"> {prompt}")
        print(result.final_output)
        print("-" * 60)


if __name__ == "__main__":
    main()
