"""Tests for deep research agent configuration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_browser_use.research.machine import ResearchMachine


@pytest.mark.anyio
async def test_research_machine_passes_use_vision_setting():
    """Research searches should honor settings.agent.use_vision."""
    mock_agent = MagicMock()
    mock_result = MagicMock()
    mock_result.final_result.return_value = "Summary"
    mock_result.history = []
    mock_agent.run = AsyncMock(return_value=mock_result)

    machine = ResearchMachine(
        topic="Test topic",
        max_searches=1,
        save_path=None,
        llm=MagicMock(),
        browser_profile=MagicMock(),
    )

    with (
        patch("mcp_server_browser_use.research.machine.Agent", return_value=mock_agent) as agent_class,
        patch("mcp_server_browser_use.research.machine.settings.agent.use_vision", False),
    ):
        result = await machine._execute_search("test query")

    call_kwargs = agent_class.call_args[1]
    assert call_kwargs["use_vision"] is False
    assert result.summary == "Summary"
