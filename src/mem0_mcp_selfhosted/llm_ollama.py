"""Custom Ollama LLM provider that restores tool-calling support.

mem0ai PR #3241 (Aug 5, 2025) accidentally removed tool-call passing and
parsing from OllamaLLM during a refactor. The original PR #1596 (Aug 2, 2024)
properly supported tool calling — issue #3222 (Jul 25, 2025) proves it worked
11 days before the regression.

This subclass overrides only the two broken methods:
- generate_response(): passes tools to client.chat()
- _parse_response(): parses response.message.tool_calls

Everything else (config handling, client creation) is inherited from OllamaLLM.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from mem0.llms.ollama import OllamaLLM

logger = logging.getLogger(__name__)


class OllamaToolLLM(OllamaLLM):
    """Ollama LLM with restored tool-calling support for graph entity extraction."""

    def _parse_response(self, response, tools):
        """Parse response with proper tool_calls extraction.

        Handles both modern Ollama SDK ToolCall objects and legacy dict format.
        """
        # Extract content from response (handles both dict and object)
        if isinstance(response, dict):
            content = response["message"]["content"]
        else:
            content = response.message.content

        if tools:
            processed_response: Dict = {
                "content": content,
                "tool_calls": [],
            }

            # Extract tool_calls from response
            tool_calls_data = None
            if isinstance(response, dict):
                tool_calls_data = response.get("message", {}).get("tool_calls")
            elif hasattr(response, "message") and hasattr(response.message, "tool_calls"):
                tool_calls_data = response.message.tool_calls

            if tool_calls_data:
                for tc in tool_calls_data:
                    if isinstance(tc, dict):
                        # Legacy dict format: {"function": {"name": ..., "arguments": ...}}
                        processed_response["tool_calls"].append({
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        })
                    else:
                        # Modern ToolCall object: tc.function.name, tc.function.arguments
                        processed_response["tool_calls"].append({
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        })

            return processed_response
        else:
            return content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: List[Dict] | None = None,
        tool_choice: str = "auto",  # Accepted for API compat; Ollama has no equivalent
        **kwargs,
    ):
        """Generate a response with proper tool passing to Ollama.

        Restores the `if tools: params["tools"] = tools` line removed in
        upstream PR #3241.
        """
        # Copy messages to avoid mutating the caller's list (upstream bug:
        # repeated calls would accumulate "Please respond with valid JSON only.")
        messages = [dict(m) for m in messages]

        params = {
            "model": self.config.model,
            "messages": messages,
        }

        # Handle JSON response format (preserved from upstream)
        if response_format and response_format.get("type") == "json_object":
            params["format"] = "json"
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nPlease respond with valid JSON only."
            else:
                messages.append({"role": "user", "content": "Please respond with valid JSON only."})

        # Add options for Ollama
        options = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        params["options"] = options

        # CRITICAL FIX: Pass tools to Ollama (removed in upstream PR #3241)
        if tools:
            params["tools"] = tools

        response = self.client.chat(**params)
        return self._parse_response(response, tools)
