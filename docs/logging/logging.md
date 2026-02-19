# Agent 全链路追踪日志

## 概述

AgentTracer 记录 v4 StateGraph 每次查询的完整执行链路，用于调试和性能分析。

## 目录结构

```
logger/
├── traces/                    # JSON 追踪文件
│   └── {timestamp}-{hash}.json
├── eval/                      # 评估结果
│   └── {testset}_{timestamp}.json
└── backup/                    # 自动归档的旧日志
    └── {timestamp}/
```

## Trace JSON 结构

```json
{
  "trace_id": "20260218-173532-7e52a1",
  "timestamp": "2026-02-18T17:35:32Z",
  "query": "用户问题",
  "config": {
    "version": "v4-langgraph",
    "max_iterations": 10,
    "max_tool_calls": 15
  },
  "routing": {
    "strategy": "single",
    "description": "v4 LangGraph StateGraph ReAct"
  },
  "query_events": [{
    "qe_id": "7e52a1",
    "question": "用户问题",
    "tool_calls": [
      {
        "tool_name": "lookup_knowledge",
        "input": {"entity": "伊涅芙"},
        "output": "## 实体信息：伊涅芙\n...",
        "duration_ms": 850
      }
    ],
    "answer": "润色后的回答..."
  }],
  "golden_answer": "标准答案 (eval 模式下存在)",
  "final_grading": {
    "faithfulness": 1.0,
    "completeness": 1.0,
    "total": 1.0,
    "duration_ms": 9403
  },
  "final_response": "原始回答 (润色前)",
  "humanized_response": "润色后回答",
  "passed": true,
  "total_duration_ms": 53450
}
```

## 使用方法

### 查看最新追踪

```bash
# 最新 trace
ls -t logger/traces/*.json | head -1 | xargs cat | jq .

# 摘要
cat logger/traces/*.json | jq '{query, passed, total_duration_ms}'
```

### 分析工具调用

```bash
# 列出所有工具调用
cat logger/traces/*.json | jq '.query_events[].tool_calls[] | {tool_name, input}'

# 统计工具使用频次
cat logger/traces/*.json | jq '.query_events[].tool_calls[].tool_name' | sort | uniq -c
```

### 查看评分

```bash
cat logger/traces/*.json | jq '.final_grading | {faithfulness, completeness, total}'
```

### 对比润色效果

```bash
# 查看润色前后差异
cat logger/traces/*.json | jq '{
  raw_len: (.final_response | length),
  humanized_len: (.humanized_response | length),
  identical: (.final_response == .humanized_response)
}'
```

### 性能分析

```bash
cat logger/traces/*.json | jq '{
  total_ms: .total_duration_ms,
  tool_calls: [.query_events[].tool_calls[].duration_ms] | add,
  grading_ms: .final_grading.duration_ms
}'
```

## 追踪生命周期

v4 StateGraph 追踪流程：

1. **start_trace**: `run()` 开始时调用，记录配置
2. **log_routing**: 记录路由策略 ("v4 LangGraph StateGraph ReAct")
3. **start_query_event**: 记录查询开始
4. **log_query_event_tool_call**: solve_tools 节点中每次工具执行后调用
5. **end_query_event**: graph 完成后调用，记录最终回答
6. **log_final_grading**: grade 节点中评分后调用
7. **end_trace**: `run_with_grading()` 完成后调用，保存 JSON

## 评估日志

`run_eval.py` 生成的评估结果存储在 `logger/eval/`：

```bash
# 查看评估结果
cat logger/eval/*.json | jq '.results[] | {id, passed, scores: .grading}'
```

旧日志在每次评估开始时自动归档到 `logger/backup/{timestamp}/`。
