# 智能会议纪要生成助手 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建一个命令行工具，通过三步处理流程将会议口水稿转换为简洁的会议纪要。

**Architecture:** 采用模块化设计，LLM调用、RAG检索、处理流程分离。使用OpenAI兼容接口调用本地vllm部署的qwen3-30b模型，三步处理流程串联执行，每步输出中间结果文件。

**Tech Stack:** Python 3.10+, openai库（兼容接口）, argparse（命令行解析）

---

## Task 1: 项目初始化与配置模块

**Files:**
- Create: `requirements.txt`
- Create: `config.py`

**Step 1: 创建依赖文件**

```python
# requirements.txt
openai>=1.0.0
```

**Step 2: 创建配置模块**

```python
# config.py
"""配置管理模块"""

API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-30b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0
```

**Step 3: 验证配置模块可导入**

运行: `python -c "import config; print(config.API_BASE_URL)"`
预期输出: `http://localhost:8000/v1`

**Step 4: 提交**

```bash
git add requirements.txt config.py
git commit -m "feat: add project config and dependencies"
```

---

## Task 2: LLM客户端模块

**Files:**
- Create: `llm_client.py`

**Step 1: 创建LLM客户端模块**

```python
# llm_client.py
"""LLM API调用封装"""
from openai import OpenAI
import config


class LLMClient:
    def __init__(
        self,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None
    ):
        self.client = OpenAI(
            base_url=config.API_BASE_URL,
            api_key="dummy"  # vllm不需要真实key
        )
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.DEFAULT_MAX_TOKENS
        self.top_p = top_p if top_p is not None else config.DEFAULT_TOP_P

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """调用模型生成回复"""
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
        return response.choices[0].message.content
```

**Step 2: 验证模块可导入**

运行: `python -c "from llm_client import LLMClient; print('OK')"`
预期输出: `OK`

**Step 3: 提交**

```bash
git add llm_client.py
git commit -m "feat: add LLM client module"
```

---

## Task 3: RAG模块（打桩实现）

**Files:**
- Create: `rag.py`

**Step 1: 创建RAG模块**

```python
# rag.py
"""RAG检索模块（打桩实现）"""

# 固定的RAG检索结果
RAG_RESULT = "zs: 张三 ls: 李四 ww: 王五 ai: 人工智能 ml: 机器学习 nlp: 自然语言处理"


def get_rag_result(query: str = "") -> str:
    """
    获取RAG检索结果（当前为打桩实现）

    Args:
        query: 检索查询（当前未使用）

    Returns:
        固定的RAG检索结果字符串
    """
    return RAG_RESULT
```

**Step 2: 验证模块可导入**

运行: `python -c "from rag import get_rag_result; print(get_rag_result())"`
预期输出: `zs: 张三 ls: 李四 ww: 王五 ai: 人工智能 ml: 机器学习 nlp: 自然语言处理`

**Step 3: 提交**

```bash
git add rag.py
git commit -m "feat: add RAG module with stub implementation"
```

---

## Task 4: 处理器模块

**Files:**
- Create: `processor.py`

**Step 1: 创建处理器模块**

```python
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
        """
        第一步：纠正人名和专有名词

        Args:
            text: 原始口水稿
            rag_result: RAG检索结果，如未提供则自动获取

        Returns:
            纠正后的文本
        """
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
        """
        第二步：删除语气词和重复字词

        Args:
            text: 第一步纠正后的文本

        Returns:
            清理后的文本
        """
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
        """
        第三步：生成会议纪要

        Args:
            text: 第二步清理后的文本

        Returns:
            会议纪要
        """
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
        """
        执行完整的处理流程

        Args:
            text: 原始口水稿
            rag_result: RAG检索结果，如未提供则自动获取
            output_callback: 中间结果输出回调函数，接收(step_name, content)

        Returns:
            (纠正后文本, 清理后文本, 会议纪要)
        """
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
```

**Step 2: 验证模块可导入**

运行: `python -c "from processor import Processor; print('OK')"`
预期输出: `OK`

**Step 3: 提交**

```bash
git add processor.py
git commit -m "feat: add processor module with three-step pipeline"
```

---

## Task 5: 主程序入口

**Files:**
- Create: `main.py`

**Step 1: 创建主程序**

```python
# main.py
"""智能会议纪要生成助手 - 命令行入口"""
import argparse
import sys
from pathlib import Path
from llm_client import LLMClient
from processor import Processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="智能会议纪要生成助手 - 将会议口水稿转换为简洁的会议纪要"
    )
    parser.add_argument(
        "input",
        help="输入文件路径，或使用 '-' 从标准输入读取"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="输出目录（默认与输入文件同目录）"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出（打印每步中间结果）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="模型温度参数（默认0.0）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大生成长度（默认2048）"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="核采样参数（默认1.0）"
    )
    return parser.parse_args()


def read_input(input_path: str) -> tuple[str, Path]:
    """读取输入内容，返回(内容, 输入文件路径)"""
    if input_path == "-":
        content = sys.stdin.read()
        return content, Path("stdin")
    else:
        path = Path(input_path)
        if not path.exists():
            print(f"错误：输入文件不存在: {input_path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8"), path


def get_output_path(input_path: Path, output_dir: str, suffix: str) -> Path:
    """生成输出文件路径"""
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = input_path.parent

    stem = input_path.stem
    return out_dir / f"{stem}_{suffix}.txt"


def main():
    args = parse_args()

    # 读取输入
    content, input_path = read_input(args.input)

    # 初始化LLM客户端和处理器
    try:
        llm_client = LLMClient(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p
        )
        processor = Processor(llm_client, verbose=args.verbose)
    except Exception as e:
        print(f"错误：无法连接到LLM服务: {e}", file=sys.stderr)
        sys.exit(1)

    # 定义输出回调
    def output_callback(step: str, text: str):
        if args.verbose:
            print(f"\n=== {step} ===\n", file=sys.stderr)
            print(text, file=sys.stderr)

    # 执行处理流程
    try:
        corrected, cleaned, summary = processor.process(
            content,
            output_callback=output_callback
        )
    except Exception as e:
        print(f"错误：处理失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 写入输出文件
    try:
        # 纠正结果
        corrected_path = get_output_path(input_path, args.output_dir, "corrected")
        corrected_path.write_text(corrected, encoding="utf-8")
        print(f"纠正结果: {corrected_path}")

        # 清理结果
        cleaned_path = get_output_path(input_path, args.output_dir, "cleaned")
        cleaned_path.write_text(cleaned, encoding="utf-8")
        print(f"清理结果: {cleaned_path}")

        # 最终纪要
        summary_path = get_output_path(input_path, args.output_dir, "summary")
        summary_path.write_text(summary, encoding="utf-8")
        print(f"会议纪要: {summary_path}")

    except PermissionError:
        print("错误：无写入权限", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 2: 验证命令行帮助**

运行: `python main.py --help`
预期输出: 显示帮助信息，包含所有参数说明

**Step 3: 提交**

```bash
git add main.py
git commit -m "feat: add CLI entry point with full argument support"
```

---

## Task 6: 集成测试

**Files:**
- Create: `test_input.txt`

**Step 1: 创建测试输入文件**

```
# test_input.txt
今天开会讨论一下我们的ai项目。首先zs同学来介绍一下背景。

嗯，好的，这个ai项目呢，主要是做nlp相关的应用。我们打算用ml的技术来实现。

好的，ls同学你来说说计划吧。

啊，好的，那个我们计划分三个阶段。第一阶段是调研，第二阶段是开发，第三阶段是测试。

嗯嗯，好的，那就这样定下来吧。
```

**Step 2: 运行基本测试**

运行: `python main.py test_input.txt -v`
预期: 无报错，生成三个输出文件，控制台显示中间结果

**Step 3: 检查输出文件**

运行: `ls test_input_*.txt`
预期: 显示三个文件
- `test_input_corrected.txt`
- `test_input_cleaned.txt`
- `test_input_summary.txt`

**Step 4: 提交**

```bash
git add test_input.txt
git commit -m "test: add sample input file for integration testing"
```

---

## 执行顺序总结

1. Task 1: 项目初始化与配置模块
2. Task 2: LLM客户端模块
3. Task 3: RAG模块
4. Task 4: 处理器模块
5. Task 5: 主程序入口
6. Task 6: 集成测试