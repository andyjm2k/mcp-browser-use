"""Tests for OpenAI-compatible structured-output sanitizing adapters."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from mcp_server_browser_use.llm_openai_compat import SanitizingChatOpenAI, sanitize_structured_json_text


class ExampleOutput(BaseModel):
    foo: str


def _mock_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
    )


def test_sanitize_structured_json_text_strips_think_wrapper():
    wrapped = '<think>\nreasoning\n</think>\n{"foo":"bar"}'
    assert sanitize_structured_json_text(wrapped) == '{"foo":"bar"}'


@pytest.mark.asyncio
async def test_sanitizing_chat_openai_parses_wrapped_json():
    llm = SanitizingChatOpenAI(model="MiniMax-M2.7", api_key="test-key", base_url="https://api.minimax.io/v1")
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_mock_response('<think>\nreasoning\n</think>\n{"foo":"bar"}'))

    with patch.object(llm, "get_client", return_value=mock_client):
        result = await llm.ainvoke([], output_format=ExampleOutput)

    assert result.completion == ExampleOutput(foo="bar")
