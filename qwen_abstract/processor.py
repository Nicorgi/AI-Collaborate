# processor.py
"""三步处理流程的核心逻辑"""
from typing import Callable, Optional
from llm_client import LLMClient
from rag import get_rag_result


class Processor:
    def __init__(self, llm_client: LLMClient, verbose: bool = False):
        self.llm = llm_client
        self.verbose = verbose

    def step1_correct(self, text: str, rag_result: str = None) -> str:
        """第一步：纠正人名和专有名词"""
        if rag_result is None:
            rag_result = get_rag_result(text)

        system_prompt = """你是一个文本纠正助手。你的任务是根据提供的参考信息，纠正文本中的人名和专有名词。

规则：
1. 仅纠正人名和专有名词，不做其他修改
2. 严格保持原文的格式、换行、标点符号不变
3. 如果参考信息中没有相关纠正，保持原文不变
4. 不要添加或删除任何内容，只做替换"""

        user_prompt = f"""参考信息：
{rag_result}

请纠正以下文本中的人名和专有名词：

{text}"""

        return self.llm.chat(system_prompt, user_prompt)

    def step2_clean(self, text: str) -> str:
        """第二步：删除语气词和重复字词"""
        system_prompt = """你是一个文本清理助手。你的任务是删除文本中的语气词和重复字词。

规则：
1. 删除语气词，如：嗯、啊、呃、这个、那个、就是、然后、所以呢等
2. 删除重复的字词，如连续重复的词
3. 严格保持原文的格式、换行不变
4. 删除后语句要通顺，但不做其他修改
5. 不要添加任何新内容"""

        user_prompt = f"""请删除以下文本中的语气词和重复字词：

{text}"""

        return self.llm.chat(system_prompt, user_prompt)

    def step3_summarize(self, text: str) -> str:
        """第三步：生成会议纪要"""
        system_prompt = """你是一个会议纪要生成助手。你的任务是将会议记录转换为简洁准确的会议纪要。

要求：
1. 提取会议的核心内容、决议和行动项
2. 语言简洁明了，去除冗余
3. 按照逻辑顺序组织内容
4. 使用规范的商务写作风格"""

        user_prompt = f"""请根据以下会议记录生成会议纪要：

{text}"""

        return self.llm.chat(system_prompt, user_prompt)

    def process(
        self,
        text: str,
        rag_result: str = None,
        output_callback: Optional[Callable[[str, str], None]] = None
    ) -> tuple[str, str, str]:
        """执行完整的处理流程"""
        if rag_result is None:
            rag_result = get_rag_result(text)

        # 第一步：纠正
        corrected = self.step1_correct(text, rag_result)
        if output_callback:
            output_callback("corrected", corrected)

        # 第二步：清理
        cleaned = self.step2_clean(corrected)
        if output_callback:
            output_callback("cleaned", cleaned)

        # 第三步：生成纪要
        summary = self.step3_summarize(cleaned)
        if output_callback:
            output_callback("summary", summary)

        return corrected, cleaned, summary