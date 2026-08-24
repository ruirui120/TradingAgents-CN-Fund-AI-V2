"""Minimal DeepSeek client used by the standalone fund project."""

from __future__ import annotations

import os


class DeepSeekChatClient:
    """Small adapter around DeepSeek's OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        model: str = "deepseek-chat",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1400,
        timeout: float = 180,
        max_retries: int = 2,
    ):
        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY；请通过环境变量注入，不要提交到 Git。")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装项目依赖。") from exc

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url
            or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, messages: list[tuple[str, str]]):
        normalized = [
            {
                "role": "user" if role in {"human", "user"} else "system",
                "content": content,
            }
            for role, content in messages
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=normalized,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek 未返回可用内容。")
        return type("DeepSeekResponse", (), {"content": content.strip()})()
