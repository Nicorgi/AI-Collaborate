# llm_client.py
"""LLM API调用封装（基于 LangChain）"""
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import config


class LLMClient:
    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        self.base_url = config.API_BASE_URL
        self.model_name = config.MODEL_NAME
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.DEFAULT_MAX_TOKENS
        self.top_p = top_p if top_p is not None else config.DEFAULT_TOP_P

        # OpenAI兼容接口通常仍要求传入api_key；本项目默认使用占位符。
        self.client = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.base_url,
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            model_kwargs={"top_p": self.top_p},
            timeout=300
        )

    def _normalize_content(self, content: object) -> str:
        """将 LangChain 返回内容规范化为字符串。"""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)

        return str(content)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """调用模型生成回复。

        Raises:
            RuntimeError: API调用失败时抛出
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt + "\n/no_think")
        ]

        try:
            response = self.client.invoke(messages)
            raw_content = self._normalize_content(response.content).strip()

            if not raw_content:
                raise RuntimeError("API返回空内容")

            # 提取 </think> 之后的内容
            if "</think>" in raw_content:
                return raw_content.split("</think>")[-1].strip()

            return raw_content
        except Exception as e:
            raise RuntimeError(f"API调用失败: {e}") from e
