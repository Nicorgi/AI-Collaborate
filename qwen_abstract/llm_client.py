# llm_client.py
"""LLM API调用封装"""
from typing import Optional
from openai import OpenAI, APIError
import config


class LLMClient:
    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        self.client = OpenAI(
            base_url=config.API_BASE_URL,
            api_key="dummy"  # vllm不需要真实key
        )
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.DEFAULT_MAX_TOKENS
        self.top_p = top_p if top_p is not None else config.DEFAULT_TOP_P

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """调用模型生成回复

        Raises:
            RuntimeError: API调用失败时抛出
        """
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )

            if not response.choices:
                raise RuntimeError("API返回空响应")

            content = response.choices[0].message.content
            if content is None:
                raise RuntimeError("API返回空内容")

            return content

        except APIError as e:
            raise RuntimeError(f"API调用失败: {e}") from e