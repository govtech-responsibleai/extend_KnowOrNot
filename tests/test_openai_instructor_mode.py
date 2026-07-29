"""Tests for the configurable Instructor mode on the OpenAI-compatible client.

PlatformAI exposes Bedrock Claude through an OpenAI-compatible endpoint.
Instructor's TOOLS_STRICT mode adds ``strict: true`` to the tool definition,
which Bedrock Claude rejects. These tests pin down the ability to select a
different Instructor mode (e.g. TOOLS) while preserving the strict default for
native OpenAI users.
"""

from typing import cast
from unittest.mock import patch

import instructor
from pydantic import BaseModel

from src.knowornot import KnowOrNot
from src.knowornot.SyncLLMClient.openai_client import SyncOpenAIClient


class WeatherResponse(BaseModel):
    temperature: float
    condition: str


@patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
def test_add_openai_defaults_to_tools_strict(mock_prompt):
    """add_openai() must keep the strict-output default for native OpenAI users."""
    mock_prompt.return_value = "Hello"
    kon = KnowOrNot()
    kon.add_openai(
        api_key="sk-test",
        default_model="gpt-4o",
        default_embedding_model="text-embedding-3-small",
    )

    client = cast(SyncOpenAIClient, kon.default_sync_client)
    assert client.instructor_client.mode == instructor.Mode.TOOLS_STRICT


@patch("src.knowornot.SyncLLMClient.openai_client.instructor.from_openai")
@patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
def test_add_openai_forwards_custom_mode_to_from_openai(mock_prompt, mock_from_openai):
    """Passing instructor.Mode.TOOLS must reach instructor.from_openai()."""
    mock_prompt.return_value = "Hello"
    kon = KnowOrNot()
    kon.add_openai(
        api_key="sk-test",
        default_model="gpt-4o",
        default_embedding_model="text-embedding-3-small",
        instructor_mode=instructor.Mode.TOOLS,
    )

    assert mock_from_openai.call_args.kwargs["mode"] == instructor.Mode.TOOLS


@patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
def test_custom_mode_works_with_custom_base_url(mock_prompt):
    """The custom mode must be honoured when a custom base_url is used (PlatformAI)."""
    mock_prompt.return_value = "Hello"
    kon = KnowOrNot()
    kon.add_openai(
        api_key="platform-key",
        base_url="https://platform.example.com/v1",
        default_model="bedrock.claude-opus-4-7",
        default_embedding_model="text-embedding-3-small",
        instructor_mode=instructor.Mode.TOOLS,
    )

    client = cast(SyncOpenAIClient, kon.default_sync_client)
    assert client.instructor_client.mode == instructor.Mode.TOOLS
    assert str(client.client.base_url) == "https://platform.example.com/v1/"


@patch("src.knowornot.SyncLLMClient.SyncLLMClient.prompt")
def test_non_strict_mode_does_not_emit_strict_in_tool_schema(mock_prompt):
    """Regression: non-strict mode must not add `strict` to the tool definition.

    Bedrock Claude rejects `tools.0.custom.strict: Extra inputs are not permitted`,
    so the tool schema generated under the selected mode must omit `strict`.
    """
    from instructor.process_response import handle_response_model

    mock_prompt.return_value = "Hello"
    kon = KnowOrNot()
    kon.add_openai(
        api_key="platform-key",
        base_url="https://platform.example.com/v1",
        default_model="bedrock.claude-opus-4-7",
        default_embedding_model="text-embedding-3-small",
        instructor_mode=instructor.Mode.TOOLS,
    )

    client = cast(SyncOpenAIClient, kon.default_sync_client)
    _, kwargs = handle_response_model(
        response_model=WeatherResponse,
        mode=client.instructor_client.mode,
        messages=[{"role": "user", "content": "hi"}],
    )
    tool_function = kwargs["tools"][0]["function"]
    assert "strict" not in tool_function
