# MiniMax-M3 固定 10 题 Table 1 最终结果（v25）

## 最终结果

两组结果使用相同固定 10 题、MiniMax-M3 三票裁判和同一版评测代码。Baseline 来源为
`runs/MiniMax-M3/search-baseline-v12`，指标目录为 `metrics/MiniMax-M3/search-baseline-v15`；
RAG 主体来源为最终引用清洗后的 `citation-rag-v20`，增加逐 claim 证据核验的技术背景事实后得到
`citation-rag-v25`。

| Test Model / System | Reference Precision | Reference Recall | Ref Num | Cited Match Rate | Cited Count | Non-cited Factual Acc | Non-cited Count |
|---|---:|---:|---:|---:|---:|---:|---:|
| M3 search baseline | 0.2283 | 0.01444 | 10.6 | 82.99% | 29.0 | 33.33% (3/9) | 0.9 |
| M3 citation-graph RAG v25 | **0.2708** | **0.01957** | 9.1 | **91.53%** | 20.5 | **100.00% (39/39)** | **3.9** |

Non-cited Factual Accuracy 按论文定义使用全部无引用事实的正确数除以总评测数，而不是把空任务记为
0%。Baseline 的逐语句总体结果是 3/9；RAG 是 39/39。逐任务宏平均另保存在 summary 中：Baseline
只对 4 个非空任务计算为 42.50%，RAG 的 10 个任务均非空且为 100%。

## 相对变化

| 指标 | Baseline | RAG v25 | 变化 |
|---|---:|---:|---:|
| Reference precision | 0.2283 | 0.2708 | +0.0425（+18.6%） |
| Reference recall | 0.01444 | 0.01957 | +0.00514（+35.6%） |
| Gold reference hits | 23 | 26 | +3 |
| Cited match rate | 82.99% | 91.53% | +8.54 pp |
| Cited count | 29.0 | 20.5 | -8.5（-29.3%） |
| Non-cited factual accuracy | 33.33% | 100.00% | +66.67 pp |
| Non-cited count | 0.9 | 3.9 | +3.0 |

## 本轮实现

1. RAG 主体仍要求论文、作者、方法、结果和比较等研究性主张使用逐句 URL 引用。
2. 新增受控 non-cited 通道：从检索论文生成技术定义或机制类候选，每条只匹配词义最相关的 3 篇本地
   学术证据，独立 M3 三票必须全票支持才允许进入报告。
3. 确定性删除无 URL 的年份、论文题名、作者归属等文献元数据，避免把未经引用的 bibliographic
   claim 混入 non-cited 集合。
4. 修复 non-cited 评测证据合并顺序：最多保留 3 条报告本地学术证据，再补 2 条免费网页检索证据。
   旧实现先放 5 条网页结果再截断，导致本地证据实际被全部丢弃。
5. 空 non-cited 集合现在输出 JSON `null` / 表格 N/A，不再以 0% 表示；CLI 同样可安全显示 N/A。

## 与论文 Table 1 的限制

论文使用 100 个任务和 Gemini Pro/Flash 共 6 票，本实验是固定 10 题和 MiniMax-M3 3 票，因此不能把
绝对值视为严格复现或宣称超过论文模型。RAG 的生成前事实门与最终评测都使用 M3，可能存在同模型裁判
偏好；毕业论文应把它明确写成 verifier-guided factuality contribution，并在条件允许时增加异构裁判或
人工抽查。当前最可靠结论是：在相同 10 题和相同评测器下，RAG 相对 baseline 同时改善了 Table 1 的
Reference Precision、Reference Recall、Cited Match Rate 和 Non-cited Factual Accuracy。

