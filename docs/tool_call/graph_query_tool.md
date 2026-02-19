# 检索工具文档

> Agent v4 的 7 个 KG + Vector 检索工具

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent (LangGraph StateGraph)                  │
├─────────────────────────────────────────────────────────────────┤
│  lookup_   │ find_      │ get_     │ trace_    │ compare_  │    │
│  knowledge │ connection │ timeline │ causality │ entities  │    │
│            │            │          │           │           │    │
│  explore_subgraph              │ search_memory              │    │
├────────────────────────────────┴────────────────────────────────┤
│                   GraphSearcher          │  HybridSearcher      │
│                   (Neo4j Cypher)         │  (Qdrant + BM25)     │
├──────────────────────────────────────────┴──────────────────────┤
│              _resolve_canonical_name (别名解析)                   │
└─────────────────────────────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
             ┌──────────┐                   ┌──────────┐
             │  Neo4j   │                   │  Qdrant  │
             │ (7687)   │                   │ (6333)   │
             └──────────┘                   └──────────┘
```

所有工具使用 `@tool` 装饰器定义，LangChain 自动生成工具 schema。

---

## 1. lookup_knowledge

**用途**：查询实体属性/直接关系

**参数**:
- `entity` (str): 实体名称（支持别名）
- `relation` (str, 可选): 关系类型过滤

**调用链**:
```
lookup_knowledge(entity, relation)
  └── GraphSearcher.search(entity, relation, limit=25)
        ├── _resolve_canonical_name(entity)
        └── Cypher: MATCH (a {name})-[r]-(b) RETURN ...
```

**适用问题**: "X是谁？"、"X的属性？"、"X的朋友有谁？"

---

## 2. find_connection

**用途**：查找两实体最短路径

**参数**:
- `entity1` (str): 起点实体
- `entity2` (str): 终点实体

**调用链**:
```
find_connection(entity1, entity2)
  └── GraphSearcher.get_path_between(entity1, entity2)
        └── Cypher: shortestPath(...) [max 4 hops]
```

**适用问题**: "X和Y什么关系？"、"X怎么认识Y？"

---

## 3. get_timeline

**用途**：追踪实体关系的时间线演变

**参数**:
- `entity` (str): 主实体
- `target` (str, 可选): 特定关系对象过滤

**调用链**:
```
get_timeline(entity, target)
  └── GraphSearcher.search_history(entity, target)
        └── Cypher: MATCH ... ORDER BY chapter, task_id ASC
```

**适用问题**: "X的经历？"、"X和Y的关系如何发展？"

---

## 4. trace_causality

**用途**：追踪事件/行为的因果链

**参数**:
- `entity` (str): 实体名称
- `event_type` (str, 可选): 事件类型过滤

**调用链**:
```
trace_causality(entity, event_type)
  └── GraphSearcher.get_major_events(entity, event_type, limit=20)
        └── Cypher: MATCH (c)-[:EXPERIENCES]->(e:MajorEvent) ...
```

**适用问题**: "为什么X会导致Y？"、"X经历了什么重大事件？"

---

## 5. compare_entities

**用途**：对比两个实体的属性和关系

**参数**:
- `entity1` (str): 第一个实体
- `entity2` (str): 第二个实体

**调用链**:
```
compare_entities(entity1, entity2)
  └── 分别调用 GraphSearcher.search() 获取两实体信息
```

**适用问题**: "X和Y有什么不同？"、"X和Y的身份对比"

---

## 6. explore_subgraph

**用途**：探索实体周围的子图结构

**参数**:
- `entity` (str): 中心实体
- `depth` (int, 可选): 探索深度

**调用链**:
```
explore_subgraph(entity, depth)
  └── GraphSearcher 多层搜索
```

**适用问题**: "X周围有哪些势力？"、"X的社会关系网"

---

## 7. search_memory

**用途**：三信号混合搜索故事原文

**参数**:
- `query` (str): 搜索查询
- `top_k` (int, 可选): 返回结果数

**三信号融合**:
1. **Dense**: BGE-M3 1024-dim 向量相似度
2. **Sparse**: BGE-M3 稀疏词法向量
3. **BM25**: jieba 分词 + BM25 关键词匹配

```
search_memory(query, top_k)
  └── HybridSearcher
        ├── dense_search (Qdrant, BGE-M3)
        ├── sparse_search (Qdrant, BGE-M3 sparse)
        ├── bm25_search (jieba tokenization)
        └── reciprocal_rank_fusion → rerank → top_k
```

**适用问题**: "描述某场景"、"X说了什么？"、需要故事原文的问题

---

## 工具选择指南

| 问题类型 | 推荐工具 | 说明 |
|---------|---------|------|
| "X是谁？" | `lookup_knowledge` | 基本信息 |
| "X的Y属性？" | `lookup_knowledge` | 属性查询 |
| "X和Y什么关系？" | `find_connection` | 最短路径 |
| "X的经历？" | `get_timeline` | 时间线 |
| "为什么X会Y？" | `trace_causality` | 因果链 |
| "X和Y有何不同？" | `compare_entities` | 对比 |
| "X周围有什么？" | `explore_subgraph` | 子图探索 |
| "描述某场景" | `search_memory` | 原文搜索 |
| 深度/细节问题 | `lookup_knowledge` → `search_memory` | 先 KG 后 Vector |

**KG-First 策略**: 优先使用知识图谱工具定位实体，再用 `search_memory` 获取原文细节。

---

## 核心机制：别名解析

所有 Neo4j 工具通过 `_resolve_canonical_name()` 进行别名解析：

```
输入: "火神" → Fulltext Search → 匹配 aliases → 输出: "玛薇卡"
```

别名来源: `src/common/config/aliases/` 目录下的 JSON 文件。

---

## 源码位置

| 文件 | 功能 |
|------|------|
| `src/agent/tools/lookup_knowledge.py` | lookup_knowledge |
| `src/agent/tools/find_connection.py` | find_connection |
| `src/agent/tools/get_timeline.py` | get_timeline |
| `src/agent/tools/trace_causality.py` | trace_causality |
| `src/agent/tools/compare_entities.py` | compare_entities |
| `src/agent/tools/explore_subgraph.py` | explore_subgraph |
| `src/agent/tools/search_memory.py` | search_memory (三信号混合搜索) |
| `src/common/graph/searcher.py` | GraphSearcher (Neo4j 查询) |
| `src/common/vector/searcher.py` | HybridSearcher (Qdrant + BM25) |
