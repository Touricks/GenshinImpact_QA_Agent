# Genshin Story QA System

> Knowledge Graph + Vector 混合检索的原神剧情问答 Agent

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green.svg)](https://langchain-ai.github.io/langgraph/)

## 项目概述

针对原神游戏剧情的检索增强问答 Agent，解决"剧情太长记不住"的痛点。当前覆盖区域：**Nod-Krai（挪德卡莱）**。

**核心特性**:
- **Graph-Vector 互补架构**: Neo4j 知识图谱（结构化事实）+ Qdrant 向量库（语义原文）
- **LangGraph StateGraph 工作流**: ReAct 循环 → 答案润色 → 三维度自评
- **7 检索工具**: 6 个 KG 工具 + 1 个三信号混合搜索（dense + sparse + BM25）
- **中文优化**: BGE-M3 多语言 Embedding（1024-dim dense + sparse lexical）+ jieba BM25
- **别名自动解析**: 支持"少女" → "Columbina"、"火神" → "玛薇卡"等别名归一化

## 系统架构

```
┌───────────────────────────────────────────────────────────────┐
│                      Web UI (Streamlit)                        │
├───────────────────────────────────────────────────────────────┤
│              AgentV4Workflow (LangGraph StateGraph)            │
│                                                               │
│  START → solve_llm → [should_continue?]                       │
│                        ├─ solve_tools → solve_llm  (工具循环)  │
│                        ├─ force_answer → humanize  (达到上限)  │
│                        └─ humanize → grade → END   (正常完成)  │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    7 Retrieval Tools                      │ │
│  │  KG: lookup_knowledge | find_connection | get_timeline   │ │
│  │      trace_causality  | compare_entities| explore_subgraph│ │
│  │  Vector: search_memory (dense + sparse + BM25 fusion)    │ │
│  └──────┬───────────────────────────────────┬───────────────┘ │
├─────────┼───────────────────────────────────┼─────────────────┤
│         │          Common Layer              │                 │
│  ┌──────▼──────┐  ┌────────────────┐  ┌─────▼──────┐         │
│  │ GraphSearcher│  │ AliasResolver  │  │  Hybrid    │         │
│  │  (Cypher)   │  │(aliases.json)  │  │  Searcher  │         │
│  └──────┬──────┘  └────────────────┘  └─────┬──────┘         │
├─────────┼───────────────────────────────────┼─────────────────┤
│         ▼                                   ▼                 │
│  ┌─────────────┐                     ┌────────────┐          │
│  │    Neo4j    │                     │   Qdrant   │          │
│  │  230 nodes  │                     │ 666 points │          │
│  │  365 edges  │                     │ hybrid vec │          │
│  └─────────────┘                     └────────────┘          │
├───────────────────────────────────────────────────────────────┤
│                       Model Layer                             │
│  ┌──────────────┐  ┌─────────────────────────────────────┐   │
│  │ Gemini 2.5   │  │ BAAI/bge-m3 (1024d dense + sparse) │   │
│  │   Flash      │  │ + jieba BM25                        │   │
│  └──────────────┘  └─────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

## 检索工具

| # | 工具 | 数据源 | 用途 | 示例问题 |
|---|------|--------|------|----------|
| 1 | `lookup_knowledge` | Neo4j | 实体属性 + 直接关系 | "少女是谁？" |
| 2 | `find_connection` | Neo4j | 两实体间最短路径 | "恰斯卡和旅行者什么关系？" |
| 3 | `get_timeline` | Neo4j | 实体事件时间线 | "月髓相关的事件" |
| 4 | `trace_causality` | Neo4j | 因果/动机链追踪 | "木偶为何对月髓感兴趣？" |
| 5 | `compare_entities` | Neo4j | 结构化实体对比 | "多托雷和桑多涅有什么不同？" |
| 6 | `explore_subgraph` | Neo4j | BFS 子图展开 | "愚人众的阵营分化" |
| 7 | `search_memory` | Qdrant | 三信号混合搜索 | "玛薇卡说了什么？" |

**KG-First 策略**: 优先使用知识图谱工具定位实体和关系，再用 `search_memory` 获取原文细节。

## 三维度自评体系

Agent 对每次回答自动进行质量评估：

| 维度 | 评估内容 | 阈值 |
|------|----------|------|
| **Faithfulness** | 回答声明是否有工具证据支持 | ≥ 0.7 |
| **Completeness** | 工具返回信息的利用程度 | ≥ 0.5 |
| **Relevance** | 回答是否切中问题 | ≥ 0.7 |

## 知识图谱

### 节点类型 (230)

| Type | Count | 属性模板 |
|------|-------|----------|
| Character | 72 | affiliation, role, goal, fate |
| Event | 49 | agent, target, cause, effect, legacy, time_order |
| Item | 39 | creator, function, location, status |
| Place | 28 | political_status, significance, inhabitants |
| Faction | 21 | leader, base, purpose, status |
| Concept | 21 | origin, nature, related_system |

### 关系类型 (365)

| Relation | Count | 语义 |
|----------|-------|------|
| INVOLVED_IN | 74 | 实体参与事件 |
| LEADS_TO | 73 | 因果/使能/依赖 |
| LOCATED_AT | 53 | 空间位置 |
| MEMBER_OF | 48 | 成员归属 |
| OPPOSED_TO | 38 | 对立敌对 |
| MOTIVATED_BY | 26 | 行为动机 |
| CREATED_BY | 23 | 创造关系 |
| ORIGINATES_FROM | 18 | 起源来源 |
| TEMPORAL_BEFORE | 12 | 事件时间顺序 |

## 快速开始

### 环境要求

- Python 3.14+
- Docker (Qdrant + Neo4j)
- Apple Silicon (MPS) 或 CUDA GPU（Embedding 模型加速，可选）

### 1. 克隆 & 安装

```bash
git clone <repo-url>
cd AmberProject

# 创建虚拟环境并安装依赖
python -m venv .venv
.venv/bin/pip install -r src/requirements.txt
```

### 2. 启动数据库

```bash
docker compose up -d
```

这将启动：
- **Qdrant**: localhost:6333（向量数据库）
- **Neo4j**: localhost:7687（图数据库），Browser: localhost:7474

### 3. 配置

复制 `.env.example` 为 `.env` 并填入 API Key：

```bash
cp .env.example .env
# 编辑 .env，至少填入:
#   GEMINI_API_KEY=your_key
```

### 4. 导入数据快照

项目附带预导出的 Nod-Krai 数据快照（Neo4j 230 节点 + Qdrant 666 向量），无需从头构建：

```bash
.venv/bin/python -m src.scripts.cli_export import Data/exports/2026-02-18_neo4j_qdrant/
```

### 5. 运行

```bash
# 单次提问
.venv/bin/python -m src.scripts.run_agent "少女是谁？"

# 单次 + 三维度评分
.venv/bin/python -m src.scripts.run_agent -g "霜月之子崇拜的神祇是谁？"

# 交互模式
.venv/bin/python -m src.scripts.run_agent -i

# Web UI
.venv/bin/streamlit run src/ui/streamlit_app.py
```

## 评估

### 数据集

41 道 Nod-Krai 剧情问答题（`TestData/cleandata/Data_only_nodkrai/dataset.json`），按难度分为 Tier 1-5。

### 运行评估

```bash
# 3 题快速验证
.venv/bin/python -m src.scripts.run_eval --testset smoke

# 7 题 Tier-1
.venv/bin/python -m src.scripts.run_eval --testset basic --workers 3

# 自选题目
.venv/bin/python -m src.scripts.run_eval --ids NK-01 NK-07 NK-15

# 全量 41 题
.venv/bin/python -m src.scripts.run_eval --testset full --workers 1
```

### 准确度评估（LLM-as-Judge）

```bash
.venv/bin/python evaluation/scripts/eval_accuracy.py --from-eval logger/eval/*.json
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph StateGraph |
| LLM | Google Gemini (primary), Anthropic Claude / OpenAI (fallback) |
| Embedding | BAAI/bge-m3 (1024-dim dense + sparse lexical) |
| 中文分词 | jieba (BM25) |
| 向量数据库 | Qdrant |
| 图数据库 | Neo4j 5 Community + APOC |
| 框架 | LangChain Core, LlamaIndex (pipeline) |
| 数据模型 | Pydantic v2 |
| UI | Streamlit |

## 项目结构

```
AmberProject/
├── src/
│   ├── agent/                  # Agent 核心
│   │   ├── agent.py            # GenshinRetrievalAgent 入口
│   │   ├── workflow_v4.py      # AgentV4Workflow (LangGraph StateGraph)
│   │   ├── llm_factory.py      # LLM 创建工厂 (Gemini/Claude/OpenAI)
│   │   ├── grader.py           # 三维度评分 (Faithfulness/Completeness/Relevance)
│   │   ├── combiner.py         # 答案润色 (Humanizer)
│   │   ├── prompts.py          # System prompt + 工具决策树
│   │   ├── tracer.py           # 全链路追踪
│   │   └── tools/              # 7 个检索工具
│   ├── common/                 # 共享基础设施
│   │   ├── alias_resolver.py   # 别名解析 (aliases.json → canonical)
│   │   ├── config/             # 配置 + 别名数据
│   │   ├── graph/              # Neo4j 接口 (GraphSearcher)
│   │   ├── models/             # 数据模型 (Entity, Relationship, Chunk)
│   │   └── vector/             # Qdrant 接口 (Indexer, M3 Embedder, Sparse)
│   ├── pipeline/               # 数据处理管道
│   │   ├── pipeline.py         # 主管道编排
│   │   ├── chunker.py          # 语义分块
│   │   ├── llm_kg_extractor.py # LLM KG 提取
│   │   ├── graph_builder.py    # Neo4j 写入
│   │   └── reranker.py         # 重排序
│   └── scripts/                # CLI 入口
│       ├── run_agent.py        # Agent CLI (单次/交互/评分)
│       ├── run_eval.py         # 并行评估
│       ├── cli_export.py       # 数据快照导入/导出
│       ├── cli_alias.py        # 别名管理
│       ├── cli_graph.py        # 图操作
│       └── cli_vector.py       # 向量操作
├── Data/
│   └── exports/                # 预导出数据快照 (Neo4j + Qdrant)
├── TestData/cleandata/         # 评估数据集 (41 题 + testsets)
├── evaluation/                 # 准确度评估 (LLM-as-Judge)
├── docs/                       # 详细文档
│   ├── PRD.md                  # 产品需求文档
│   ├── agent/architecture.md   # v4 架构详解
│   ├── agent/agent.md          # Agent API
│   ├── tool_call/              # 工具文档
│   ├── data_Ingestion/         # 数据管道文档
│   └── logging/                # 日志追踪
└── docker-compose.yml          # Qdrant + Neo4j
```

## 文档

详细文档见 [`docs/`](docs/README.md)：

| 文档 | 内容 |
|------|------|
| [PRD](docs/PRD.md) | 产品需求文档 |
| [Agent 架构](docs/agent/architecture.md) | v4 LangGraph StateGraph 详解 |
| [Agent API](docs/agent/agent.md) | Agent 调用方法 |
| [检索工具](docs/tool_call/graph_query_tool.md) | 7 个工具详细文档 |
| [数据管道](docs/data_Ingestion/overview.md) | 双数据库灌入流程 |

## License

MIT License
