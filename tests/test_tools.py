"""Tests for the function-tool builders and the get_agent_memory_tools factory."""

from __future__ import annotations

import json

import pytest
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext

from agent_memory_openai_agents_sdk import DEFAULT_OPERATIONS, get_agent_memory_tools


def _tool_context() -> ToolContext:
    """Build a minimal ToolContext for invoking a tool directly in tests."""
    return ToolContext(
        context=None,
        tool_name="test",
        tool_call_id="call_1",
        tool_arguments="{}",
    )


async def _invoke(tool, **arguments) -> str:
    return await tool.on_invoke_tool(_tool_context(), json.dumps(arguments))


def test_default_tools_cover_all_operations(fake_client):
    tools = get_agent_memory_tools(fake_client)
    assert [tool.name for tool in tools] == list(DEFAULT_OPERATIONS)


def test_include_selects_and_orders_tools(fake_client):
    tools = get_agent_memory_tools(fake_client, include=("recall", "remember"))
    assert [tool.name for tool in tools] == ["recall", "remember"]


def test_unknown_operation_raises(fake_client):
    with pytest.raises(ValueError, match="Unknown Agent Memory operation"):
        get_agent_memory_tools(fake_client, include=("teleport",))


def test_recall_tool_schema_has_query(fake_client):
    (recall,) = get_agent_memory_tools(fake_client, include=("recall",))
    properties = recall.params_json_schema["properties"]
    assert "query" in properties
    assert "limit" in properties


async def test_remember_then_recall_round_trip(fake_client):
    remember, recall = get_agent_memory_tools(
        fake_client, session_id="s1", include=("remember", "recall")
    )

    await _invoke(remember, content="The sky is blue")
    result = await _invoke(recall, query="sky")

    assert "The sky is blue" in result


async def test_scope_is_threaded_to_client(fake_client):
    (remember,) = get_agent_memory_tools(
        fake_client, session_id="s1", user_id="u1", include=("remember",)
    )

    await _invoke(remember, content="hello")

    stored = fake_client.stored[0]
    assert stored["scope"].session_id == "s1"
    assert stored["scope"].user_id == "u1"


async def test_recall_is_scoped(fake_client):
    (remember_s1,) = get_agent_memory_tools(
        fake_client, session_id="s1", include=("remember",)
    )
    (recall_s2,) = get_agent_memory_tools(
        fake_client, session_id="s2", include=("recall",)
    )

    await _invoke(remember_s1, content="secret in s1")
    result = await _invoke(recall_s2, query="secret")

    assert result == ""
