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