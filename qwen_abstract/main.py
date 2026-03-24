#!/usr/bin/env python
# main.py
"""命令行入口程序"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from docx import Document
from llm_client import LLMClient
from processor import Processor


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="qwen_abstract",
        description="会议记录处理工具：纠正专有名词、清理语气词、生成会议纪要"
    )

    parser.add_argument(
        "input",
        help="输入文件路径，使用 '-' 从 stdin 读取"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认与输入文件同目录，stdin 模式下默认当前目录）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细模式，输出中间结果到 stderr"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="模型温度参数（默认使用配置值）"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大生成 token 数（默认使用配置值）"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p 采样参数（默认使用配置值）"
    )

    return parser.parse_args()


def read_docx(file_path: Path) -> str:
    """
    读取 docx 文件内容

    Args:
        file_path: docx 文件路径

    Returns:
        文件中的文本内容（按段落合并）
    """
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def read_input(input_path: str) -> tuple[str, Optional[Path]]:
    """
    读取输入内容

    Args:
        input_path: 输入路径，'-' 表示 stdin

    Returns:
        (文本内容, 输入文件路径或 None)
    """
    if input_path == "-":
        return sys.stdin.read(), None
    else:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"输入文件不存在: {path}")
        if not path.is_file():
            raise ValueError(f"输入路径不是文件: {path}")

        # 根据文件扩展名选择读取方式
        if path.suffix.lower() == ".docx":
            return read_docx(path), path
        else:
            return path.read_text(encoding="utf-8"), path


def get_output_paths(
    input_path: Optional[Path],
    output_dir: Optional[Path]
) -> tuple[Path, Path, Path]:
    """
    确定输出文件路径

    Args:
        input_path: 输入文件路径（stdin 时为 None）
        output_dir: 用户指定的输出目录

    Returns:
        (corrected_path, cleaned_path, summary_path)
    """
    if output_dir is None:
        if input_path is None:
            output_dir = Path.cwd()
        else:
            output_dir = input_path.parent

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定文件名 stem
    if input_path is None:
        stem = "stdin"
    else:
        stem = input_path.stem

    corrected_path = output_dir / f"{stem}_corrected.txt"
    cleaned_path = output_dir / f"{stem}_cleaned.txt"
    summary_path = output_dir / f"{stem}_summary.txt"

    return corrected_path, cleaned_path, summary_path


def verbose_callback(stage: str, content: str) -> None:
    """详细模式回调，输出到 stderr"""
    stage_names = {
        "corrected": "=== 纠正结果 ===",
        "cleaned": "=== 清理结果 ===",
        "summary": "=== 会议纪要 ==="
    }
    print(f"\n{stage_names.get(stage, stage)}", file=sys.stderr)
    print(content, file=sys.stderr)


def main() -> int:
    """主函数入口"""
    args = parse_args()

    try:
        # 读取输入
        text, input_path = read_input(args.input)

        if not text.strip():
            print("错误: 输入内容为空", file=sys.stderr)
            return 1

        # 创建 LLM 客户端
        try:
            llm_client = LLMClient(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p
            )
        except Exception as e:
            print(f"错误: 初始化 LLM 客户端失败 - {e}", file=sys.stderr)
            return 1

        # 创建处理器
        processor = Processor(llm_client, verbose=args.verbose)

        # 设置回调
        callback = verbose_callback if args.verbose else None

        # 执行处理
        try:
            corrected, cleaned, summary = processor.process(text, output_callback=callback)
        except RuntimeError as e:
            print(f"错误: 处理失败 - {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"错误: API 连接失败 - {e}", file=sys.stderr)
            return 1

        # 确定输出路径
        corrected_path, cleaned_path, summary_path = get_output_paths(
            input_path, args.output_dir
        )

        # 写入输出文件
        try:
            corrected_path.write_text(corrected, encoding="utf-8")
            cleaned_path.write_text(cleaned, encoding="utf-8")
            summary_path.write_text(summary, encoding="utf-8")
        except PermissionError as e:
            print(f"错误: 权限不足，无法写入文件 - {e}", file=sys.stderr)
            return 1
        except OSError as e:
            print(f"错误: 写入文件失败 - {e}", file=sys.stderr)
            return 1

        # 输出结果路径
        print(f"纠正结果: {corrected_path}")
        print(f"清理结果: {cleaned_path}")
        print(f"会议纪要: {summary_path}")

        return 0

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n操作已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())