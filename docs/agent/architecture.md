# Agent v4 Architecture

> LangGraph StateGraph ReAct 工作流

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    AgentV4Workflow (StateGraph)                   │
│                                                                   │
│  ┌───────────┐     ┌──────────────┐     ┌───────────────┐       │
│  │ solve_llm │ ←──→│ solve_tools  │     │ force_answer  │       │
│  │ (Gemini   │     │ (7 tools)    │     │ (fallback)    │       │
│  │  3 Pro)   │     └──────────────┘     └───────┬───────┘       │
│  └─────┬─────┘                                   │               │
│        │ should_continue                         │               │
│        ▼                                         ▼               │
│  ┌───────────┐                           ┌───────────────┐       │
│  │ humanize  │ ◄─────────────────────────│               │       │
│  │ (润色)    │                            └───────────────┘       │
│  └─────┬─────┘                                                   │
│        ▼                                                         │
│  ┌───────────┐                                                   │
│  │  grade    │  → Faithfulness / Completeness → Total           │
│  │(golden_a) │                                                   │
│  └─────┬─────┘                                                   │
│        ▼                                                         │
│      END → AgentResponse                                        │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Neo4j   │        │  Qdrant  │        │  Gemini  │
   │ 知识图谱  │        │ 向量库   │        │   LLM    │
   └──────────┘        └──────────┘        └──────────┘
```

---

## StateGraph 定义

### AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # LangChain 消息列表
    question: str                            # 用户问题
    qe_id: str                               # 查询事件 ID

    tool_calls: Annotated[list[ToolCallRecord], add]  # 工具调用记录 (append)
    total_tool_call_count: int               # 工具调用总数
    solve_iterations: int                    # LLM 调用轮次
    solve_start_ts: float                    # 求解开始时间

    raw_answer: str                          # 润色前原始回答
    humanized_answer: str                    # 润色后回答
    golden_answer: str                         # golden answer (eval 模式)
    grading: Optional[GradingResult]         # golden_answer 评分
```

### Graph 结构

```
START → solve_llm → [should_continue?]
                      ├─ "solve_tools"  → solve_tools → solve_llm  (循环)
                      ├─ "force_answer" → force_answer → humanize  (达到上限)
                      └─ "humanize"     → humanize → grade → END   (正常完成)
```

### 路由逻辑 (should_continue)

| 条件 | 路由目标 |
|------|----------|
| LLM 未请求工具调用 | → humanize (直接润色) |
| 工具调用次数 ≥ 上限 | → force_answer (强制回答) |
| 迭代轮次 ≥ 上限 | → force_answer (强制回答) |
| 否则 | → solve_tools (执行工具) |

---

## 节点详解

### 1. solve_llm

**职责**: 调用 REASONING_MODEL（绑定 7 个工具），生成推理和工具调用请求。

- 模型: `gemini-3-pro-preview` (可配置)
- 使用 `bind_tools()` 绑定工具 schema
- 每次调用递增 `solve_iterations`

### 2. solve_tools

**职责**: 执行 LLM 请求的工具调用。

- 遍历 `AIMessage.tool_calls`，逐个执行
- 如果中途达到工具上限，为剩余调用添加 placeholder ToolMessage
- 记录 `ToolCallRecord`（工具名、输入、输出、耗时）
- 通过 Tracer 记录每次工具调用

### 3. force_answer

**职责**: 当达到工具/迭代上限时，强制 LLM 生成文本回答。

- 为所有未响应的 tool_call 添加 placeholder ToolMessage
- 调用 `reasoning_llm.ainvoke()` （不绑定工具）强制文本输出

### 4. humanize

**职责**: 将原始回答润色为自然语言。

- 使用 GRADER_MODEL（快速模型）
- 移除推理噪声、工具痕迹、自我验证
- 提升语言自然度（添加过渡句，避免列表堆砌）
- 保留所有实体名称，不编造新事实

### 5. grade

**职责**: golden_answer 模式评分。

有 golden_answer 时，两维度通过 `asyncio.gather` 并行评估，使用 `with_structured_output` 确保格式可靠。
无 golden_answer（生产环境）时跳过评分，零 LLM 调用。

| 维度 | 评估内容 |
|------|----------|
| **Faithfulness** | Agent 回答与 golden_answer 的事实一致性（只惩罚矛盾，不惩罚额外信息） |
| **Completeness** | Agent 回答对 golden_answer 要点的覆盖率 |
| **Total** | (faithfulness + completeness) / 2 |

---

## 模型配置

| 角色 | 环境变量 | 默认模型 | 用途 |
|------|----------|----------|------|
| REASONING_MODEL | `REASONING_MODEL` | `gemini-3-pro-preview` | 主推理 + 工具编排 |
| GRADER_MODEL | `GRADER_MODEL` | `gemini-2.5-flash` | 评分 + 润色 |

LLM 创建通过 `llm_factory.create_chat_model()` 统一管理，支持 Gemini / Claude / OpenAI。

---

## 7 个检索工具

| 工具 | 数据源 | 用途 | 典型问题 |
|------|--------|------|----------|
| `lookup_knowledge` | Neo4j | 查询实体属性/直接关系 | "X是谁？" |
| `find_connection` | Neo4j | 查找两实体最短路径 | "X和Y什么关系？" |
| `get_timeline` | Neo4j | 追踪关系时间线 | "X的经历？" |
| `trace_causality` | Neo4j | 追踪因果链 | "为什么X会导致Y？" |
| `compare_entities` | Neo4j | 对比两实体属性 | "X和Y有何不同？" |
| `explore_subgraph` | Neo4j | 探索实体子图 | "X周围有什么？" |
| `search_memory` | Qdrant | 三信号混合搜索 | "描述某场景" |

所有工具使用 `@tool` 装饰器预定义，LangChain 自动处理 schema 生成和参数解析。

---

## 执行流程示例

### 问题: "伊涅芙是什么身份？"

```
1. [solve_llm] LLM 分析问题 → 调用 lookup_knowledge(entity="伊涅芙")
2. [solve_tools] 执行工具 → 返回 Neo4j 实体关系数据
3. [solve_llm] LLM 综合信息 → 生成最终文本回答 (无更多工具调用)
4. [should_continue] 无工具调用 → 路由到 humanize
5. [humanize] 润色回答 → 自然语言 (622ch → 494ch)
6. [grade] 评分 → faith=1.0, comp=1.0, total=1.0
7. [END] → AgentResponse
```

---

## 重试机制

`run()` 方法包装 graph.ainvoke() 在重试循环中：

- 最大重试: 2 次
- 重试条件: 空回答 或 可重试异常 (503, 500, unavailable)
- 每次重试使用全新 initial_state（消息重置）
- 指数退避: 1s → 2s

---

## Checkpointing

使用 LangGraph `MemorySaver` 提供检点支持：

- 每次 graph.ainvoke() 使用唯一 `thread_id`
- 支持未来扩展：断点恢复、human-in-the-loop

---

## 文件结构

```
src/agent/
├── agent.py             # GenshinRetrievalAgent 入口
├── workflow_v4.py       # AgentV4Workflow (StateGraph)
├── llm_factory.py       # LLM 创建工厂 (Gemini/Claude/OpenAI)
├── grader.py            # ProductionGraderService (golden_answer 评分)
├── combiner.py          # ResponseHumanizer (答案润色)
├── prompts.py           # AGENT_V4_SYSTEM_PROMPT + HUMANIZE_PROMPT
├── models.py            # ToolCallRecord, AgentResponse, GradingResult
├── tracer.py            # AgentTracer 全链路追踪
├── settings.py          # agent_settings 代理
└── tools/               # 7 个检索工具
    ├── lookup_knowledge.py
    ├── find_connection.py
    ├── get_timeline.py
    ├── trace_causality.py
    ├── compare_entities.py
    ├── explore_subgraph.py
    └── search_memory.py   # 三信号混合搜索
```

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [agent.md](./agent.md) | Agent API 调用方法 |
| [../tool_call/graph_query_tool.md](../tool_call/graph_query_tool.md) | 工具详细文档 |
| [../logging/logging.md](../logging/logging.md) | 追踪日志格式 |
