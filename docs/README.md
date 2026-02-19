# Genshin Story QA System - Documentation

> 文档导航页 | Documentation Hub

## Quick Links

| 文档类型 | 路径 | 说明 |
|----------|------|------|
| 产品需求 | [PRD.md](./PRD.md) | Product Requirements Document |
| Agent 架构 | [agent/architecture.md](./agent/architecture.md) | v4 LangGraph StateGraph 架构 |
| Agent API | [agent/agent.md](./agent/agent.md) | Agent 调用方法 |
| 工具文档 | [tool_call/](./tool_call/) | 7 个检索工具详解 |
| 数据管道 | [data_Ingestion/](./data_Ingestion/) | 双数据库灌入流程 |
| 日志追踪 | [logging/logging.md](./logging/logging.md) | 全链路追踪日志 |

---

## 文档结构

```
docs/
├── README.md                          # 本文件 - 导航页
├── PRD.md                             # 产品需求文档
├── agent/                             # Agent 核心文档
│   ├── architecture.md                # v4 LangGraph StateGraph 架构
│   └── agent.md                       # Agent API 调用方法
├── tool_call/                         # 检索工具
│   └── graph_query_tool.md            # KG + Vector 工具详解
├── data_Ingestion/                    # 数据灌入
│   ├── overview.md                    # 灌入架构概览
│   └── vector-incremental-pipeline.md # 向量增量管道
└── logging/                           # 日志
    └── logging.md                     # 追踪日志格式
```

---

## Agent v4 架构概览

### 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph StateGraph |
| 推理模型 | Google Gemini 3 Pro (REASONING_MODEL) |
| 评分/润色模型 | Google Gemini 2.5 Flash (GRADER_MODEL) |
| 向量数据库 | Qdrant (BGE-M3 dense+sparse + BM25) |
| 图数据库 | Neo4j 5 + APOC |
| 框架 | LangChain Core + LangGraph |

### StateGraph 流程

```
START → solve_llm → [should_continue?]
                      ├─ solve_tools → solve_llm  (工具调用循环)
                      ├─ force_answer → humanize   (达到上限)
                      └─ humanize → grade → END    (正常完成)
```

**节点说明**:
- **solve_llm**: 调用 LLM（绑定 7 个工具），生成推理 + 工具调用
- **solve_tools**: 执行工具调用，记录 ToolCallRecord，返回结果
- **force_answer**: 达到工具/迭代上限时，强制生成文本回答
- **humanize**: 润色原始回答为自然语言
- **grade**: golden_answer 评分 (Faithfulness + Completeness → Total)

### 7 个检索工具

| 工具 | 数据源 | 用途 |
|------|--------|------|
| `lookup_knowledge` | Neo4j | 查询实体属性/直接关系 |
| `find_connection` | Neo4j | 查找两实体最短路径 |
| `get_timeline` | Neo4j | 追踪关系时间线演变 |
| `trace_causality` | Neo4j | 追踪因果链 |
| `compare_entities` | Neo4j | 对比两实体属性 |
| `explore_subgraph` | Neo4j | 探索实体周围子图 |
| `search_memory` | Qdrant | 三信号混合搜索 (dense + sparse + BM25) |

### 评分体系 (golden_answer 模式)

| 维度 | 说明 |
|------|------|
| Faithfulness | Agent 回答与 golden_answer 的事实一致性 |
| Completeness | Agent 回答对 golden_answer 要点的覆盖率 |
| Total | (faithfulness + completeness) / 2 |

生产环境（无 golden_answer）：跳过评分，零 LLM 调用。

详细架构文档: [agent/architecture.md](./agent/architecture.md)

---

## 版本历史

| 日期 | 变更 |
|------|------|
| 2026-02-18 | v4 LangGraph StateGraph 架构，更新全部文档 |
| 2026-02-01 | 新增 Agent 执行流程文档 |
| 2026-01-29 | 按 Google Style 重构文档结构 |
| 2026-01-27 | 初始文档创建 |
