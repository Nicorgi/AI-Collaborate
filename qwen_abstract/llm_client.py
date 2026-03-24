# llm_client.py
"""LLM API调用封装"""
from typing import Optional
import requests
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

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """调用模型生成回复

        Raises:
            RuntimeError: API调用失败时抛出
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "/no_think"}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()

            data = response.json()

            if "choices" not in data or not data["choices"]:
                raise RuntimeError("API返回空响应")

            content = data["choices"][0]["message"]["content"].split("\n")[-1]
            if content is None:
                raise RuntimeError("API返回空内容")

            return content

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API调用失败: {e}") from e