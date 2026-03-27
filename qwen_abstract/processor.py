# processor.py
"""三步处理流程的核心逻辑"""
from typing import Callable, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from llm_client import LLMClient
from rag import get_rag_result


class Processor:
    def __init__(
        self,
        llm_client: LLMClient,
        verbose: bool = False,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

        self.llm = llm_client
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            length_function=len
        )

    def _split_text(self, text: str) -> list[str]:
        """按配置分块，空白块会被过滤。"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.splitter.split_text(text)
        return [chunk for chunk in chunks if chunk.strip()]

    def _run_chunked(
        self,
        chunks: list[str],
        chunk_handler: Callable[[str], str]
    ) -> str:
        """对每个分块执行同一阶段的模型调用并拼接结果。"""
        outputs: list[str] = []
        for chunk in chunks:
            outputs.append(chunk_handler(chunk))

        return "".join(outputs)

    @staticmethod
    def _batch_items(items: list[str], batch_size: int) -> list[list[str]]:
        """把列表拆成固定大小批次。"""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

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

        chunks = self._split_text(text)

        def invoke(chunk: str) -> str:
            user_prompt = f"""参考信息：
{rag_result}

请纠正以下文本中的人名和专有名词：

{chunk}"""
            return self.llm.chat(system_prompt, user_prompt)

        return self._run_chunked(chunks, invoke)

    def step2_clean(self, text: str) -> str:
        """第二步：删除语气词和重复字词"""
        system_prompt = """你是一个文本清理助手。你的任务是删除文本中的语气词和重复字词。

规则：
1. 删除语气词，如：嗯、啊、呃、这个、那个、就是、然后、所以呢等
2. 删除重复的字词，如连续重复的词
3. 严格保持原文的格式、换行不变
4. 删除后语句要通顺，但不做其他修改
5. 不要添加任何新内容"""

        chunks = self._split_text(text)

        def invoke(chunk: str) -> str:
            user_prompt = f"""请删除以下文本中的语气词和重复字词：

{chunk}"""
            return self.llm.chat(system_prompt, user_prompt)

        return self._run_chunked(chunks, invoke)

    def step3_summarize(self, text: str) -> str:
        """第三步：生成会议纪要"""
        map_system_prompt = """你是一个会议纪要生成助手。你的任务是将会议记录转换为简洁准确的会议纪要。

要求：
1. 提取会议的核心内容、决议和行动项
2. 语言简洁明了，去除冗余
3. 按照逻辑顺序组织内容
4. 使用规范的商务写作风格"""

        reduce_system_prompt = """你是会议纪要整合助手。请将多个分段纪要整合为一份最终纪要。

要求：
1. 合并重复信息，保留关键结论和行动项
2. 保证时间线和逻辑顺序清晰
3. 输出简洁、结构化、可直接分发"""

        chunks = self._split_text(text)
        partial_summaries: list[str] = []

        for chunk in chunks:
            user_prompt = f"""请根据以下会议记录片段生成阶段纪要：

{chunk}"""
            partial_summaries.append(self.llm.chat(map_system_prompt, user_prompt))

        # 单块直接返回，避免额外模型调用
        if len(partial_summaries) == 1:
            return partial_summaries[0]

        current = partial_summaries
        while len(current) > 1:
            merged: list[str] = []
            for batch in self._batch_items(current, batch_size=6):
                payload = "\n\n".join(
                    f"[分段纪要{i + 1}]\n{item}" for i, item in enumerate(batch)
                )
                user_prompt = f"""请将以下分段纪要整合为更完整、去重后的纪要：

{payload}"""
                merged.append(self.llm.chat(reduce_system_prompt, user_prompt))
            current = merged

        return current[0]

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
