"""OpenAI-compatible adapters for providers with non-standard structured output."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any, TypeVar, overload

from browser_use import ChatOpenAI
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion
from openai import APIConnectionError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_OPEN_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def _extract_first_json_value(text: str) -> str:
    """Return the first balanced JSON object/array found in text."""
    stack: list[str] = []
    start: int | None = None
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}

    for index, char in enumerate(text):
        if start is None:
            if char in pairs:
                start = index
                stack.append(pairs[char])
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in pairs:
            stack.append(pairs[char])
            continue

        if stack and char == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                return text[start : index + 1]

    return text.strip()


def sanitize_structured_json_text(text: str) -> str:
    """Strip common reasoning wrappers and isolate the JSON payload."""
    cleaned = text.strip()
    cleaned = _THINK_TAG_RE.sub("", cleaned).strip()
    cleaned = _CODE_FENCE_OPEN_RE.sub("", cleaned)
    cleaned = _CODE_FENCE_CLOSE_RE.sub("", cleaned).strip()
    extracted = _extract_first_json_value(cleaned)
    return extracted.strip() if extracted else cleaned


class SanitizingChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that tolerates reasoning wrappers around JSON output."""

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Invoke the model, sanitizing wrapped JSON for structured output when needed."""
        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

        try:
            model_params: dict[str, Any] = {}

            if self.temperature is not None:
                model_params["temperature"] = self.temperature

            if self.frequency_penalty is not None:
                model_params["frequency_penalty"] = self.frequency_penalty

            if self.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self.max_completion_tokens

            if self.top_p is not None:
                model_params["top_p"] = self.top_p

            if self.seed is not None:
                model_params["seed"] = self.seed

            if self.service_tier is not None:
                model_params["service_tier"] = self.service_tier

            if self.reasoning_models and any(str(model).lower() in str(self.model).lower() for model in self.reasoning_models):
                model_params["reasoning_effort"] = self.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            if output_format is None:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **model_params,
                )

                usage = self._get_usage(response)
                return ChatInvokeCompletion(
                    completion=response.choices[0].message.content or "",
                    usage=usage,
                    stop_reason=response.choices[0].finish_reason if response.choices else None,
                )

            response_format: JSONSchema = {
                "name": "agent_output",
                "strict": True,
                "schema": SchemaOptimizer.create_optimized_json_schema(
                    output_format,
                    remove_min_items=self.remove_min_items_from_schema,
                    remove_defaults=self.remove_defaults_from_schema,
                ),
            }

            if self.add_schema_to_system_prompt and openai_messages and openai_messages[0]["role"] == "system":
                schema_text = f"\n<json_schema>\n{response_format}\n</json_schema>"
                if isinstance(openai_messages[0]["content"], str):
                    openai_messages[0]["content"] += schema_text
                elif isinstance(openai_messages[0]["content"], Iterable):
                    openai_messages[0]["content"] = [
                        *openai_messages[0]["content"],
                        ChatCompletionContentPartTextParam(text=schema_text, type="text"),
                    ]

            if self.dont_force_structured_output:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **model_params,
                )
            else:
                response = await self.get_client().chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    response_format=ResponseFormatJSONSchema(json_schema=response_format, type="json_schema"),
                    **model_params,
                )

            content = response.choices[0].message.content
            if content is None:
                raise ModelProviderError(
                    message="Failed to parse structured output from model response",
                    status_code=500,
                    model=self.name,
                )

            usage = self._get_usage(response)

            try:
                parsed = output_format.model_validate_json(content)
            except ValidationError:
                sanitized = sanitize_structured_json_text(content)
                if sanitized == content:
                    raise
                logger.debug("Sanitized wrapped structured output for model %s", self.name)
                parsed = output_format.model_validate_json(sanitized)

            return ChatInvokeCompletion(
                completion=parsed,
                usage=usage,
                stop_reason=response.choices[0].finish_reason if response.choices else None,
            )

        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e
        except APIConnectionError as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
        except APIStatusError as e:
            raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
