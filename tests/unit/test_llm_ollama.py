"""Tests for llm_ollama.py — OllamaToolLLM tool-calling support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mem0.llms.ollama import OllamaLLM

from mem0_mcp_selfhosted.llm_ollama import OllamaToolLLM


@pytest.fixture
def mock_ollama_config():
    """Create a mock OllamaConfig for OllamaToolLLM."""
    config = MagicMock()
    config.model = "qwen3:14b"
    config.temperature = 0.7
    config.max_tokens = 8192
    config.top_p = 0.9
    config.ollama_base_url = "http://localhost:11434"
    return config


@pytest.fixture
def ollama_llm(mock_ollama_config):
    """Create an OllamaToolLLM with mocked client."""
    with patch.object(OllamaToolLLM, "__init__", lambda self, config=None: None):
        llm = OllamaToolLLM.__new__(OllamaToolLLM)
        llm.config = mock_ollama_config
        llm.client = MagicMock()
        return llm


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Extract entities from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array"},
                },
            },
        },
    }
]


class TestSubclass:
    def test_is_subclass_of_ollama_llm(self):
        """OllamaToolLLM inherits from upstream OllamaLLM."""
        assert issubclass(OllamaToolLLM, OllamaLLM)

    def test_isinstance_check(self, ollama_llm):
        """Instance passes isinstance check for OllamaLLM."""
        assert isinstance(ollama_llm, OllamaLLM)


class TestGenerateResponse:
    def test_tools_passed_to_client_chat(self, ollama_llm):
        """When tools are provided, they are passed to client.chat()."""
        mock_response = MagicMock()
        mock_response.message.content = "test"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            tools=SAMPLE_TOOLS,
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["tools"] == SAMPLE_TOOLS

    def test_no_tools_not_in_params(self, ollama_llm):
        """When tools is None, 'tools' key is not passed to client.chat()."""
        mock_response = MagicMock()
        mock_response.message.content = "plain response"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert "tools" not in params

    def test_json_format_adds_format_param(self, ollama_llm):
        """JSON response format adds format='json' to params."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "extract facts"}],
            response_format={"type": "json_object"},
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["format"] == "json"

    def test_non_json_object_format_ignored(self, ollama_llm):
        """Non-json_object response_format types are ignored (no format param)."""
        mock_response = MagicMock()
        mock_response.message.content = "plain text"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "text"},
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert "format" not in params

    def test_json_format_does_not_mutate_caller_messages(self, ollama_llm):
        """JSON response format must not mutate the caller's messages list."""
        mock_response = MagicMock()
        mock_response.message.content = '{"facts": []}'
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        original_content = "extract facts"
        messages = [{"role": "user", "content": original_content}]

        ollama_llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
        )

        # Caller's list and dict must be unmodified
        assert len(messages) == 1
        assert messages[0]["content"] == original_content

    def test_options_include_config_values(self, ollama_llm):
        """Options dict includes temperature, num_predict, top_p from config."""
        mock_response = MagicMock()
        mock_response.message.content = "test"
        mock_response.message.tool_calls = None
        ollama_llm.client.chat.return_value = mock_response

        ollama_llm.generate_response(
            messages=[{"role": "user", "content": "hello"}],
        )

        params = ollama_llm.client.chat.call_args.kwargs
        assert params["options"]["temperature"] == 0.7
        assert params["options"]["num_predict"] == 8192
        assert params["options"]["top_p"] == 0.9


class TestParseResponse:
    def test_tool_calls_parsed_from_toolcall_objects(self, ollama_llm):
        """Modern SDK format: ToolCall objects with .function.name/.arguments."""
        tc = MagicMock()
        tc.function.name = "extract_entities"
        tc.function.arguments = {"entities": [{"name": "Alice"}]}

        response = MagicMock()
        response.message.content = ""
        response.message.tool_calls = [tc]

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == [
            {"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}]}}
        ]

    def test_tool_calls_parsed_from_dict_format(self, ollama_llm):
        """Legacy dict format: {"function": {"name": ..., "arguments": ...}}."""
        response = {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "extract_entities",
                            "arguments": {"entities": [{"name": "Bob"}]},
                        }
                    }
                ],
            }
        }

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == [
            {"name": "extract_entities", "arguments": {"entities": [{"name": "Bob"}]}}
        ]

    def test_empty_tool_calls_when_none(self, ollama_llm):
        """When response has no tool_calls, returns empty list."""
        response = MagicMock()
        response.message.content = "I found no entities"
        response.message.tool_calls = None

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == []
        assert result["content"] == "I found no entities"

    def test_empty_tool_calls_when_empty_list(self, ollama_llm):
        """When response has empty tool_calls list, returns empty list."""
        response = MagicMock()
        response.message.content = "No entities"
        response.message.tool_calls = []

        result = ollama_llm._parse_response(response, tools=SAMPLE_TOOLS)

        assert result["tool_calls"] == []

    def test_non_tool_response_returns_string(self, ollama_llm):
        """When tools is None/empty, returns plain string content."""
        response = MagicMock()
        response.message.content = "Hello, how can I help?"

        result = ollama_llm._parse_response(response, tools=None)

        assert result == "Hello, how can I help?"

    def test_non_tool_dict_response_returns_string(self, ollama_llm):
        """Dict response without tools returns plain string."""
        response = {"message": {"content": "Plain text response"}}

        result = ollama_llm._parse_response(response, tools=None)

        assert result == "Plain text response"
