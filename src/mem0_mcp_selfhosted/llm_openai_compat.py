"""Custom OpenAI-compatible LLM provider for models that lack response_format support.

Many OpenAI-compatible APIs (doubao, some qwen deployments, older vLLM, etc.)
do not support `response_format: {"type": "json_object"}`.  mem0ai's built-in
OpenAILLM passes this parameter verbatim, causing 400 errors.

This subclass intercepts `response_format` and converts it to a prompt-based
approach (same strategy as our OllamaToolLLM), then uses `extract_json()` to
parse the result.  Everything else (client creation, config, tool handling) is
inherited from OpenAILLM.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from mem0.llms.openai import OpenAILLM
from mem0.memory.utils import extract_json

logger = logging.getLogger(__name__)


class OpenAICompatLLM(OpenAILLM):
    """OpenAI-compatible LLM that replaces response_format with prompt injection."""

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """Generate response, converting response_format to prompt injection.

        When the upstream caller requests JSON output via response_format,
        we strip the parameter and instead append a JSON instruction to the
        last user message.  The raw text response is then parsed with
        extract_json() to handle code-fenced or wrapped JSON.
        """
        needs_json = (
            response_format
            and isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        )

        # qwen3 thinking mode interferes with tool calling — disable it
        # when tools are requested and model is qwen3-based
        if tools and self._is_qwen3_thinking_model():
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"].setdefault("enable_thinking", False)

        if needs_json:
            # Copy messages to avoid mutating the caller's list
            messages = [dict(m) for m in messages]

            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just the JSON object."
            else:
                messages.append({
                    "role": "user",
                    "content": "You MUST respond with valid JSON only. No markdown, no explanation, just the JSON object.",
                })

            # Call parent WITHOUT response_format
            result = super().generate_response(
                messages=messages,
                response_format=None,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

            # extract_json handles code-fenced, wrapped, or raw JSON
            if isinstance(result, str):
                return extract_json(result)
            return result
        else:
            # No JSON format requested — pass through unchanged
            return super().generate_response(
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

    def _is_qwen3_thinking_model(self) -> bool:
        """Check if the model is qwen3 with default thinking mode enabled."""
        model = getattr(self.config, "model", "") or ""
        return model.startswith("qwen3-")
