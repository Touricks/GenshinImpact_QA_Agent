# Agent 调用方法

> v4 LangGraph StateGraph Agent API

## 概览

| 方法 | 模式 | 评分 | 追踪 | 适用场景 |
|------|------|------|------|----------|
| `run(query)` | 单次 | 内含 | 无 | 代码调用 |
| `run_with_grading(query)` | 单次 | 内含 | 保存 trace | 评估/调试 |

---

## 方法详解

### 1. `run(query) -> str`

单次查询，返回答案文本。

**参数**:
- `query` (str): 用户问题

**返回值**:
- `str`: 润色后的答案文本

**行为**:
- 执行完整 StateGraph 流程 (solve → humanize → grade)
- 不保存 trace 文件

---

### 2. `run_with_grading(query) -> Tuple[str, AgentResponse]`

带完整评分和追踪的查询。

**参数**:
- `query` (str): 用户问题

**返回值**:
- `str`: 润色后的答案文本
- `AgentResponse`: 完整响应对象

**AgentResponse 结构**:

```python
class AgentResponse(BaseModel):
    question: str              # 用户问题
    answer_text: str           # 润色后答案
    raw_answer: str            # 润色前原始答案
    tool_calls: list[ToolCallRecord]  # 工具调用记录
    grading: GradingResult     # 三维度评分
    query_count: int           # 查询次数 (v4 固定为 1)
    total_duration_ms: int     # 总耗时 (ms)
```

**GradingResult 结构**:

```python
class GradingResult(BaseModel):
    faithfulness: float      # 忠实度 [0, 1]
    completeness: float      # 完整度 [0, 1]
    answer_relevance: float  # 相关度 [0, 1]
    passed: bool             # 是否通过
```

**行为**:
- 执行完整 StateGraph 流程
- 保存 trace 到 `logger/traces/`

---

## 工厂函数

### `create_agent(session_id, model, verbose, enable_grader)`

```python
from src.agent import create_agent

agent = create_agent(
    session_id="test-001",         # 会话 ID
    model="gemini-3-pro-preview",  # 推理模型 (可选)
    verbose=True,                  # 详细日志
    enable_grader=True,            # 启用评分
)
```

Agent 使用懒初始化 — 首次调用 `run()` 时才加载 LLM 和工具。

---

## 使用示例

### 命令行

```bash
# 单次提问
.venv/bin/python -m src.scripts.run_agent "少女是谁？"

# 单次 + 三维度评分
.venv/bin/python -m src.scripts.run_agent -g "霜月之子崇拜的神祇是谁？"

# 交互模式
.venv/bin/python -m src.scripts.run_agent -i
```

### 代码调用

```python
import asyncio
from src.agent import create_agent

async def main():
    agent = create_agent()

    # 简单查询
    answer = await agent.run("伊涅芙是什么身份？")
    print(answer)

    # 带评分
    answer, result = await agent.run_with_grading("少女是如何重回世界的？")
    print(f"答案: {answer}")
    print(f"忠实度: {result.grading.faithfulness:.2f}")
    print(f"通过: {result.grading.passed}")
    print(f"工具调用: {len(result.tool_calls)}")
    print(f"耗时: {result.total_duration_ms}ms")

asyncio.run(main())
```

### 批量评估

```bash
# 3 题快速验证
.venv/bin/python -m src.scripts.run_eval --testset smoke

# 7 题 Tier-1
.venv/bin/python -m src.scripts.run_eval --testset basic --workers 3

# 自选题目
.venv/bin/python -m src.scripts.run_eval --ids NK-01 NK-07 NK-15

# 全量评估
.venv/bin/python -m src.scripts.run_eval --testset full --workers 1
```

---

## 评分体系

三维度并行评估，使用 `with_structured_output` 确保格式可靠：

| 维度 | 评估内容 | 阈值 |
|------|----------|------|
| Faithfulness | 回答声明是否有工具证据支持 | ≥ 0.7 |
| Completeness | 工具返回信息的利用程度 | ≥ 0.5 |
| Relevance | 回答是否切中问题 | ≥ 0.7 |

通过条件: 三个维度均达到阈值。
