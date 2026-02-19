"""
向量数据库工具：search_memory - 检索故事原文和对话。

此工具查询 Qdrant 向量数据库获取故事内容。
这是**唯一**返回故事原文的工具。

Supports hybrid search: BM25 (jieba) + BGE-M3 sparse (lexical) + BGE-M3 dense (1024-dim).
"""

import logging
from typing import Dict, List, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ...common.vector.indexer import VectorIndexer
from ...common.config.settings import settings
from ...common.alias_resolver import resolve as _resolve_alias, get_all_names as _get_all_names
from ...pipeline.reranker import JinaReranker

logger = logging.getLogger(__name__)


# 模块级单例，避免每次调用时重新加载模型
_indexer: Optional[VectorIndexer] = None
_reranker: Optional[JinaReranker] = None
_m3_embedder = None  # BGEM3Embedder (lazy)
_bm25_embedder = None  # ChineseBM25Embedder (lazy)


def _get_indexer() -> VectorIndexer:
    """获取或创建向量索引器单例。"""
    global _indexer
    if _indexer is None:
        _indexer = VectorIndexer()
    return _indexer


def _get_reranker() -> JinaReranker:
    """获取或创建 Jina Reranker 单例。"""
    global _reranker
    if _reranker is None:
        _reranker = JinaReranker()
    return _reranker


def _get_m3_embedder():
    """获取或创建 BGE-M3 embedder 单例。"""
    global _m3_embedder
    if _m3_embedder is None:
        from ...common.vector.m3_embedder import get_m3_embedder
        _m3_embedder = get_m3_embedder()
    return _m3_embedder


def _get_bm25_embedder():
    """获取或创建 BM25 embedder 单例。Returns None if IDF not fitted."""
    global _bm25_embedder
    if _bm25_embedder is None:
        from ...common.vector.sparse_embedder import get_bm25_embedder
        _bm25_embedder = get_bm25_embedder(auto_load=True)
    return _bm25_embedder


def _hybrid_search(
    query: str,
    filter_conditions: Optional[dict],
    target_limit: int,
    sort_by: str,
) -> List[dict]:
    """Execute hybrid three-channel search.

    Returns:
        Results from hybrid search (deduped at indexer layer by point_id).
    """
    indexer = _get_indexer()
    m3 = _get_m3_embedder()
    bm25 = _get_bm25_embedder()

    # One M3 encode → dense + lexical
    m3_output = m3.encode_single(query)
    query_dense = m3.dense_to_list(m3_output["dense"])
    query_lexical = m3.lexical_to_sparse(m3_output["lexical"])

    # BM25 sparse (may be None if not fitted)
    query_bm25 = None
    if bm25 and bm25.is_fitted:
        query_bm25 = bm25.transform(query)

    # Over-fetch for reranker: each channel gets its quota
    raw_results = indexer.search_hybrid(
        query_dense=query_dense,
        query_bm25=query_bm25,
        query_lexical=query_lexical,
        quota_dense=settings.HYBRID_QUOTA_DENSE,
        quota_bm25=settings.HYBRID_QUOTA_BM25,
        quota_lexical=settings.HYBRID_QUOTA_LEXICAL,
        filter_conditions=filter_conditions,
    )

    unique_results = raw_results

    if sort_by == "time":
        unique_results.sort(
            key=lambda x: (
                x.get("payload", {}).get("chapter_number", 0),
                x.get("payload", {}).get("event_order", 0),
            )
        )

    return unique_results


class SearchMemoryInput(BaseModel):
    """search_memory 工具输入参数。"""
    query: str = Field(description="描述要搜索的事件、对话或场景的自然语言查询")
    characters: Optional[str] = Field(default=None, description="可选的实体名过滤器，只返回提及此实体的内容，支持别名自动解析")
    sort_by: Literal["relevance", "time"] = Field(default="relevance", description="排序方式: relevance=语义相关度, time=章节事件顺序")
    limit: int = Field(default=5, le=20, description="返回结果的最大数量")


@tool(args_schema=SearchMemoryInput)
def search_memory(
    query: str,
    characters: Optional[str] = None,
    sort_by: str = "relevance",
    limit: int = 5,
) -> str:
    """搜索故事原文，获取特定剧情细节、对话或事件描述（三信号混合搜索：dense+sparse+BM25）。

    这是唯一返回故事原文的工具。使用场景：查询台词、场景描写、KG工具未覆盖的细节。
    注意：返回的原文对话可能包含其他区域的无关内容。对于结构化问题（关系、时间线、因果），KG工具的结果更准确——仅在需要具体引用/台词时使用此工具。"""
    logger.info(
        f"[Qdrant] search_memory: query={query[:50]}{'...' if len(query) > 50 else ''}, "
        f"characters={characters}, sort_by={sort_by}, limit={limit}"
    )

    # 构建过滤条件（支持别名解析 + 多名称匹配）
    filter_conditions = None
    resolved_name = None
    all_names = None
    if characters:
        all_names = _get_all_names(characters)
        resolved_name = _resolve_alias(characters)

        if len(all_names) > 1:
            filter_conditions = {"entities_mentioned": all_names}
        else:
            filter_conditions = {"entities_mentioned": resolved_name}

    reranker = _get_reranker()
    target_limit = min(limit, reranker.top_k)

    # Hybrid search (dense + sparse + BM25)
    unique_results = _hybrid_search(query, filter_conditions, target_limit, sort_by)

    # Fallback cascade: entities_mentioned → characters → query augment
    fallback_used = False
    if not unique_results and filter_conditions and resolved_name:
        # Try characters field (speaker names)
        char_filter = {"characters": all_names if len(all_names) > 1 else resolved_name}

        unique_results = _hybrid_search(query, char_filter, target_limit, sort_by)

        if not unique_results:
            # Final fallback: add entity name to query, remove filter
            logger.info("[Qdrant] Filter returned 0, falling back to semantic search")
            augmented_query = f"{resolved_name} {query}"

            unique_results = _hybrid_search(augmented_query, None, target_limit, sort_by)

            fallback_used = True

    # Rerank: pick top-k from candidates
    if len(unique_results) > target_limit:
        unique_results = reranker.rerank_with_metadata(
            query=query, results=unique_results, text_key="text", top_k=target_limit,
        )
        logger.info(f"[Reranker] reranked → {target_limit}")

    results = unique_results[:target_limit]

    logger.debug(
        f"[Qdrant] search_memory result: {len(results)} chunks "
        f"(target={target_limit})"
    )

    if not results:
        msg = f"未找到与查询 '{query}' 相关的故事内容"
        if characters:
            if resolved_name and resolved_name != characters:
                msg += f"（已过滤实体：{characters} → {resolved_name}）"
            else:
                msg += f"（已过滤实体：{characters}）"
        msg += "\n\n建议：\n"
        msg += "- 尝试更宽泛或不同的查询词。\n"
        msg += "- 移除过滤器以搜索所有内容。\n"
        msg += "- 使用 lookup_knowledge 验证实体名是否正确。"
        return msg

    # 格式化结果
    lines = [f"## 故事内容：\"{query}\""]
    if characters:
        if fallback_used:
            lines.append(f"（过滤无结果，已改用语义搜索：{resolved_name}）")
        elif all_names and len(all_names) > 1:
            names_str = " | ".join(all_names)
            lines.append(f"（已过滤实体：{characters} → [{names_str}]）")
        elif resolved_name and resolved_name != characters:
            lines.append(f"（已过滤实体：{characters} → {resolved_name}）")
        else:
            lines.append(f"（已过滤实体：{characters}）")
    lines.append(f"（排序方式：{sort_by}）")
    lines.append("")

    for i, result in enumerate(results, 1):
        payload = result["payload"]
        text = payload.get("text", "")
        chapter = payload.get("chapter_number", "?")
        task_id = payload.get("task_id", "未知")
        event_order = payload.get("event_order", 0)
        score = result.get("score", 0)
        channel = result.get("_channel", "dense")

        lines.append(f"### 结果 {i}")
        lines.append(f"**来源**: 第 {chapter} 章，任务: {task_id}，事件 #{event_order}")
        if sort_by == "relevance":
            lines.append(f"**相关度**: {score:.3f}  [channel: {channel}]")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
