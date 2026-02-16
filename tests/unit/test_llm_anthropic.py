"""Tests for llm_anthropic.py — parse_response, schema detection, extract_json."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mem0_mcp_selfhosted.llm_anthropic import (
    FACT_RETRIEVAL_SCHEMA,
    MEMORY_UPDATE_SCHEMA,
    AnthropicOATLLM,
    extract_json,
)


class TestExtractJson:
    def test_clean_json(self):
        assert extract_json('{"key": "value"}') == '{"key": "value"}'

    def test_code_fenced_json(self):
        text = '```json\n{"facts": ["a", "b"]}\n```'
        assert extract_json(text) == '{"facts": ["a", "b"]}'

    def test_code_fenced_no_lang(self):
        text = '```\n{"facts": ["a"]}\n```'
        assert extract_json(text) == '{"facts": ["a"]}'

    def test_text_prefixed_json(self):
        text = 'Here is the result:\n{"facts": ["a"]}'
        result = extract_json(text)
        assert result.startswith('{"facts"')

    def test_text_prefixed_array(self):
        text = "The output is: [1, 2, 3]"
        result = extract_json(text)
        assert result.startswith("[1,")

    def test_empty_string(self):
        assert extract_json("") == ""

    def test_unclosed_code_fence(self):
        text = '```json\n{"key": "value"}'
        result = extract_json(text)
        assert '{"key": "value"}' in result


class TestParseResponse:
    def _make_text_block(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def _make_tool_use_block(self, name: str, input_data: dict):
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.input = input_data
        return block

    def _make_response(self, blocks):
        response = MagicMock()
        response.content = blocks
        return response

    def test_single_tool_use(self):
        response = self._make_response([
            self._make_tool_use_block("extract_entities", {"entities": ["Alice"]})
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert result["tool_calls"] == [
            {"name": "extract_entities", "arguments": {"entities": ["Alice"]}}
        ]
        assert result["content"] == ""

    def test_mixed_text_and_tool(self):
        response = self._make_response([
            self._make_text_block("Analyzing..."),
            self._make_tool_use_block("extract_entities", {"entities": ["Alice"]}),
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert result["content"] == "Analyzing..."
        assert len(result["tool_calls"]) == 1

    def test_text_only_when_tools_provided(self):
        response = self._make_response([
            self._make_text_block("No entities found."),
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert result["content"] == "No entities found."
        assert result["tool_calls"] == []

    def test_multiple_tool_use_blocks(self):
        response = self._make_response([
            self._make_tool_use_block("tool_a", {"a": 1}),
            self._make_tool_use_block("tool_b", {"b": 2}),
        ])
        result = AnthropicOATLLM._parse_response(response)
        assert len(result["tool_calls"]) == 2


class TestSchemaDetection:
    def test_fact_extraction_has_system_message(self):
        """System message present → FACT_RETRIEVAL_SCHEMA."""
        llm = MagicMock(spec=AnthropicOATLLM)
        messages = [
            {"role": "system", "content": "Extract facts from the following..."},
            {"role": "user", "content": "Alice prefers TypeScript"},
        ]
        result = AnthropicOATLLM._select_schema(llm, messages)
        assert result == FACT_RETRIEVAL_SCHEMA

    def test_memory_update_no_system_message(self):
        """No system message → MEMORY_UPDATE_SCHEMA."""
        llm = MagicMock(spec=AnthropicOATLLM)
        messages = [
            {"role": "user", "content": "Update memory..."},
        ]
        result = AnthropicOATLLM._select_schema(llm, messages)
        assert result == MEMORY_UPDATE_SCHEMA
