# MiniMax-M3 自适应检索固定 10 题结果（2026-08-21）

本轮冻结系统为 `search-baseline-v14` 与 `citation-rag-v22`。两组使用相同的固定 10 题、MiniMax-M3、五次检索预算、截止日期过滤和三票 M3 statement judge。五次检索采用 3+2 自适应策略：先读取三次检索结果，再用剩余两次补充缺失的 landmark paper、benchmark、dataset 或 named method。RAG 在相同直接检索之上增加三层引用图与 grounded evidence repair。

## Table 1 对齐结果

| 系统 | Ref. P | Ref. R | Ref. count | Gold hits | Cited match | Cited count | Non-cited accuracy | Non-cited count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 baseline-v14 | 0.2945 | 0.01454 | 9.3 | 27 | 0.7391 | 23.3 | 0.6444 | 1.4（共14条） |
| M3 RAG-v22 | **0.3072** | **0.01677** | 9.1 | **28** | **0.9676** | 22.4 | **0.9550** | 4.9（共49条） |

微平均结果：baseline reference precision/recall 为 0.2903/0.01403，cited accuracy 为 0.7339，non-cited accuracy 为 0.5714（8/14）；RAG reference precision/recall 为 0.3077/0.01455，cited accuracy 为 0.9688，non-cited accuracy 为 0.9592（47/49）。

## 配对提升

- Reference precision：+0.01270（相对 +4.31%）。
- Reference recall：+0.00223（相对 +15.31%）。
- Gold reference hits：27 → 28。
- Cited match：+0.22853（相对 +30.92%）。
- Non-cited factual accuracy：+0.31056（相对 +48.19%）。
- 平均引用数略降：9.3 → 9.1；平均 cited statements 略降：23.3 → 22.4。

## 与上一轮完整结果比较

自适应检索使 baseline reference precision 从 0.2350 提升到 0.2945（+25.3%），RAG 从 0.2470 提升到 0.3072（+24.4%）。RAG cited match 从 0.9001 提升到 0.9676。当前结果首次在相同 10 题下同时表现为 RAG 高于 baseline 的 reference precision、reference recall、cited match 和 non-cited factual accuracy。

## 口径限制

论文的 non-cited fact checking 使用 Gemini 2.5 Pro 与 Flash 各三票并联网检索；本实验只能使用 MiniMax-M3 三票和 MiniMax/免费学术检索证据，因此绝对分数不是对原论文 judge 的严格复刻。最可靠的结论是相同 evaluator、题集和资源约束下的 baseline/RAG 配对差异。未引用事实为零的任务记作 N/A，不按 0% 或 100%计入宏平均。

逐题数据和聚合结果位于 `metrics/MiniMax-M3/search-baseline-v14/` 与 `metrics/MiniMax-M3/citation-rag-v22/`。
