# 智能会议纪要生成助手设计文档

## 概述

基于vllm部署的qwen3-30b模型，实现三步处理流程的会议纪要生成命令行工具。

## 1. 项目结构

```
qwen_abstract/
├── main.py              # 入口文件，命令行参数解析
├── config.py            # 配置管理（API地址、模型参数）
├── rag.py               # RAG检索模块（打桩实现）
├── processor.py         # 三步处理流程的核心逻辑
├── llm_client.py        # LLM API调用封装
└── requirements.txt     # 依赖：openai
```

## 2. 命令行接口

```bash
# 基本用法
python main.py input.txt                    # 输出到同目录，自动命名
python main.py input.txt -o custom.txt       # 指定输出目录

# 详细模式（显示中间处理结果到控制台）
python main.py input.txt -v
python main.py input.txt --verbose

# 从标准输入读取
echo "会议内容..." | python main.py -
cat meeting.txt | python main.py -

# 自定义模型参数
python main.py input.txt --temperature 0.1 --max-tokens 4096
```

**输出文件命名规则**（假设输入为 `meeting.txt`）：
- `meeting_corrected.txt` - 第一步纠正结果
- `meeting_cleaned.txt` - 第二步清理结果
- `meeting_summary.txt` - 第三步最终纪要

## 3. 三步处理流程

### 第一步：纠正人名和专有名词
- 将RAG检索结果（键值对格式）和原始口水稿组合成prompt
- 要求模型仅替换人名和专有名词，保持原有格式、换行、标点不变
- 输出：`{filename}_corrected.txt`

### 第二步：删除语气词和重复字词
- 输入第一步的结果
- 要求模型仅删除语气词（如"嗯、啊、这个"）和重复字词，保持原有格式不变
- 输出：`{filename}_cleaned.txt`

### 第三步：生成会议纪要
- 输入第二步的结果
- 生成简洁准确的会议纪要
- 输出：`{filename}_summary.txt`

## 4. 核心模块设计

### 4.1 LLM客户端 (`llm_client.py`)
- 封装OpenAI兼容接口调用
- 使用openai库，base_url指向本地vllm服务
- 默认参数：temperature=0, max_tokens=2048, top_p=1.0
- 支持命令行参数覆盖

### 4.2 RAG模块 (`rag.py`)
- 打桩实现，返回固定字符串
- 格式：`原词: 正确词` 键值对，空格分隔
- 示例：`zs: 张三 ai: 人工智能`

### 4.3 处理器 (`processor.py`)
- `step1_correct()` - 纠正人名和专有名词
- `step2_clean()` - 删除语气词和重复字词
- `step3_summarize()` - 生成会议纪要
- `process()` - 串联三步，可选输出中间结果

### 4.4 配置管理 (`config.py`)
- API_BASE_URL: `http://localhost:8000/v1`
- MODEL_NAME: `qwen3-30b`
- DEFAULT_TEMPERATURE: 0.0
- DEFAULT_MAX_TOKENS: 2048
- DEFAULT_TOP_P: 1.0

## 5. 错误处理

| 错误类型 | 处理方式 | 退出码 |
|---------|---------|-------|
| API连接失败 | 提示检查vllm服务 | 1 |
| 输入文件不存在 | 提示路径错误 | 1 |
| 模型生成失败 | 显示错误信息 | 1 |
| 输出目录无权限 | 提示权限问题 | 1 |

所有错误信息输出到stderr，不影响正常输出的stdout。

## 6. 依赖

```
openai>=1.0.0
```