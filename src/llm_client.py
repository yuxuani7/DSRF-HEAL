import time
import traceback
from typing import Any, Dict, Iterable, Optional

from src.types import LLMCallError, LLMCallResult, LLMConfig, Message


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Missing dependency: openai") from exc

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def build_params(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(self.config.default_params)
        if overrides:
            merged.update(overrides)
        return merged

    def chat(
        self,
        messages: Iterable[Message],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> LLMCallResult:
        params = self.build_params(overrides)
        stream = bool(params.get("stream", False))

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=list(messages),
                **params,
            )
            if stream:
                return self._handle_stream_response(response, start)
            return self._handle_non_stream_response(response, start)
        except Exception as exc:  # pylint: disable=broad-except
            latency_ms = int((time.perf_counter() - start) * 1000)
            return LLMCallResult(
                text="",
                finish_reason=None,
                latency_ms=latency_ms,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                raw_chunk_count=0,
                error=LLMCallError(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                ),
            )

    def _handle_stream_response(self, response: Any, start: float) -> LLMCallResult:
        text_parts = []
        finish_reason: Optional[str] = None
        chunk_count = 0
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

        for chunk in response:
            chunk_count += 1
            if getattr(chunk, "usage", None) is not None:
                prompt_tokens, completion_tokens, total_tokens = self._extract_usage(chunk.usage)
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if content:
                    text_parts.append(self._stringify_delta_content(content))
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMCallResult(
            text="".join(text_parts),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_chunk_count=chunk_count,
            error=None,
        )

    def _handle_non_stream_response(self, response: Any, start: float) -> LLMCallResult:
        text = ""
        finish_reason = None
        choices = getattr(response, "choices", None) or []
        if choices:
            first = choices[0]
            finish_reason = getattr(first, "finish_reason", None)
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                if content:
                    text = self._stringify_delta_content(content)

        prompt_tokens, completion_tokens, total_tokens = self._extract_usage(
            getattr(response, "usage", None)
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMCallResult(
            text=text,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_chunk_count=1,
            error=None,
        )

    def _extract_usage(self, usage_obj: Any) -> tuple:
        if usage_obj is None:
            return None, None, None
        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        completion_tokens = getattr(usage_obj, "completion_tokens", None)
        total_tokens = getattr(usage_obj, "total_tokens", None)
        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            return prompt_tokens, completion_tokens, total_tokens
        if isinstance(usage_obj, dict):
            return (
                usage_obj.get("prompt_tokens"),
                usage_obj.get("completion_tokens"),
                usage_obj.get("total_tokens"),
            )
        return None, None, None

    def _stringify_delta_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(getattr(item, "text", "")))
            return "".join(parts)
        return str(content)

