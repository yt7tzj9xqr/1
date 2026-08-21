# MiniMax-M3 固定 10 题最终实验报告（2026-08-21）

## 结论

当前固定 10 题已经完整跑通免费检索 baseline 与三层 citation-graph RAG。最终生成实验为 `search-baseline-v10` 和 `citation-rag-v17`；随后只对完全相同的报告与来源池执行确定性的 URL 归位，得到最终评测目录 `search-baseline-v12` 和 `citation-rag-v20`。该离线步骤不调用模型、不重新检索、不增删引用，只把模型输出的 `事实句。 URL`、`事实句。 [URL]` 和 `事实句。 (URL)` 规范为引用属于前一句的格式。

在最终正确解析口径下，RAG 相对 baseline 同时提高 reference precision、reference recall 和 cited match，并减少引用语句总量：reference precision 从 0.2350 提升到 0.2470，reference recall 从 0.01298 提升到 0.01970，cited match 从 0.8674 提升到 0.9001，平均 cited statement 从 28.6 降到 22.1。RAG 检索池比 baseline 多命中 5 篇 gold 文献，最终报告多命中 1 篇。

Non-cited accuracy 不能声称提高：baseline 只剩 4 条无引用事实，micro accuracy 为 0.75；RAG 的无引用事实为 0 条，因此 accuracy 的分母为 0，数学上应记为 N/A。当前聚合器为了兼容原仓库把空任务记为 0，所以 summary 中会显示 raw macro=0；这不表示事实准确率为 0%。RAG 在这一维度可支持的结论是“消除了检测到的无引用事实语句”，而不是“提高了 non-cited accuracy”。

## 最终配对结果

两组都使用相同固定 10 题、MiniMax-M3 生成器、三票 M3 judge、免费检索源、缓存与引用解析规则。报告均 10/10 成功，无空报告。

| 系统 | Ref. P | Ref. R | Ref. count | Gold hits | Cited match | Cited micro | Cited count | Non-cited count | Non-cited micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 search-baseline-v12 | 0.2350 | 0.01298 | 10.8 | 25 | 0.8674 | 0.8706 | 28.6 | 0.4（共 4 条） | 0.7500（3 个非空任务） |
| M3 citation-rag-v20 | **0.2470** | **0.01970** | 10.1 | **26** | **0.9001** | **0.9140** | **22.1** | **0.0（共 0 条）** | N/A |

| 变化 | 数值 |
|---|---:|
| Reference precision | +0.01205（+5.1%） |
| Reference recall | +0.00673（+51.8%） |
| Gold reference 命中 | +1 |
| Cited match | +0.03275（+3.8%） |
| Cited micro | +0.04340 |
| Cited statement count | -6.5（-22.7%） |
| 检测到的 non-cited statement | 4 → 0 |

## 与论文 Page 7 Table 1 对照

| 系统 | Ref. P | Ref. R | Ref. count | Cited match | Cited count | Non-cited accuracy | Non-cited count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 论文 Gemini 2.5 Flash | 0.237 | 0.012 | 5.47 | 0.4488 | 12.10 | 0.9852 | 11.50 |
| 论文 Gemini 2.5 Pro | 0.269 | 0.010 | 4.27 | 0.5924 | 6.58 | 0.9608 | 9.35 |
| 论文 o3 | 0.299 | 0.031 | 12.26 | 0.3143 | 16.16 | 0.8222 | 11.51 |
| 论文 Claude 4 Sonnet | 0.337 | 0.021 | 6.74 | 0.7367 | 14.93 | 0.9264 | 17.07 |
| M3 search-baseline-v12 | 0.235 | 0.01298 | 10.8 | 0.8674 | 28.6 | 0.7500* | 0.4 |
| M3 citation-rag-v20 | 0.247 | 0.01970 | 10.1 | 0.9001 | 22.1 | N/A | 0.0 |

`*` Baseline non-cited accuracy 只有 4 条语句，样本不足。论文使用 100 题、Gemini Pro/Flash 六票验证 non-cited；本实验固定 10 题且 cited/non-cited 均使用 M3 三票代理，因此绝对值不能作为严格复现。最可信的是同一实现下 baseline 与 RAG 的配对差异。

这 10 题平均每题有 192.5 条 gold reference，reference recall 的分母远大于生成报告合理引用数。RAG 的 reference precision 已高于论文 Gemini Flash，recall 高于 Gemini Flash/Pro；cited match 高于 Table 1 全部行，但由于 judge 不同，不能宣称模型整体超过论文系统。

## 检索池与引用图审计

| 系统 | Pool papers | Pool gold | 最终使用的 pool papers | 最终 gold |
|---|---:|---:|---:|---:|
| Baseline v10/v12 | 159 | 35 | 108 | 25 |
| RAG v17/v20 | 362 | 40 | 101 | 26 |

RAG 的三层图平均保存 36.2 个节点，其中平均 20.2 个节点来自 depth≥1；baseline 每题约 16 篇候选。引用图新增 203 个节点和 5 个 gold pool 命中，最终写作阶段转化为 1 个额外 gold 引用。说明 graph expansion 确实提高召回，但下一步主要瓶颈是从图池到最终引用的重排转化率，而不是继续无上限扩大图。

当前实现只允许 depth 0/1 进入写作 evidence，depth 2/3 仅用于遍历；写作选择至少保留约 65% 直接搜索证据，避免高被引但偏题的深层节点挤掉任务中心文献。

## 免费实现与贡献点

1. **免费 baseline 检索与抓取。** M3 规划 5 个学术查询并发执行，MiniMax Search 为主要网页检索入口；OpenAlex、Semantic Scholar、arXiv 和 Crossref 只做免费结构化补充。匿名接口 429 时快速熔断，不阻塞主流程。
2. **三层 citation-graph RAG。** 从直接检索 seed 的结构化引用边分层扩展，按主题相关性、引用影响、深度与来源类型筛选，深层节点仅作 traversal bridge；最终通过直接证据保留与图证据补充形成写作上下文。
3. **本地正文读取与缓存。** arXiv PDF 优先由本地 `pdftotext` 读取，网页正文、搜索、模型响应和评测结果写入 SQLite 缓存。没有使用 Firecrawl、SerpAPI 或新增付费 API Key。
4. **grounded writing 与质量门。** 每个事实句绑定一个 URL；证据修复会删除不受支持的复合句。初稿、修复稿和 citation cleanup 后均检查最低词数与来源覆盖，并为 M3 length exhaustion 提供 compact/focused recovery。
5. **引用解析修复。** 将句号后的方括号、圆括号和裸 URL 归位到前一句。该修复同时提高 baseline 与 RAG，避免通过压低 baseline 制造贡献；离线重处理脚本可复现实验且不会重新调用模型。

## 关键失败实验与原因

- `search-baseline-v5` 的 3+2 反馈搜索只有 16 个 gold 命中，低于完整五查询路径。反馈查询过窄、重复题名和匿名学术源限流抵消了 agentic feedback 的理论优势，因此只保留为消融。
- v6/v7 出现 111 词、0 URL 或 149 词、2 URL 的静默短输出。原因是 M3 有时以 `stop` 返回不完整结论，旧代码只处理 `finish_reason=length`。初稿与修复稿质量门已经修复。
- 旧 statement 评测把 `事实句。 URL` 拆成无引用 claim 与单独 URL，导致 baseline/RAG cited 同时偏低并产生大量假 non-cited。三种 URL 归位规则修复后，`2306` baseline cited 从 0.286 提升到 0.929，证明这是解析错误而非搜索证据差。
- 同配置 RAG v17/v18 的 non-cited 波动较大，根本原因是无引用语句样本极少且引用格式解析不完整。最终 v20 正确归位后为 0 条，因此该 accuracy 不再具备统计意义。

## 下一步

固定 10 题已满足 reference recall ≥0.012 和 cited match ≥0.75；RAG 同时改善 reference precision、recall、cited match，并减少引用语句与无引用事实数量。扩展剩余 20 题之前需要冻结以下口径：

- Table 1 主表对 non-cited accuracy 在 count=0 时标为 N/A，而不是 0 或 1；同时始终报告 count、micro 与非空任务数。
- 30 题必须使用当前单一 commit、同一缓存与相同 URL 归位规则，不能混用早期 v10/v18 指标。
- M2.7 全实验应在 M3 的 30 题配置完全冻结后运行，避免模型差异与工程修复混杂。
- 对 `2204`、`2205`、`2310` 三个 reference=0 的题做检索 query/title error analysis，但不能读取目标 survey bibliography 作为主实验检索源。

## 可复现文件

- 最终 baseline：`metrics/MiniMax-M3/search-baseline-v12/reference/` 与 `metrics/MiniMax-M3/search-baseline-v12/full/`
- 最终 RAG：`metrics/MiniMax-M3/citation-rag-v20/reference/` 与 `metrics/MiniMax-M3/citation-rag-v20/full/`
- 生成来源：`runs/MiniMax-M3/search-baseline-v10/` 与 `runs/MiniMax-M3/citation-rag-v17/`（默认不提交 Git）
- 离线引用归位：`scripts/reprocess_citations.py`
- 检索审计：`scripts/audit_retrieval.py`
- 历史消融：`metrics/MiniMax-M3/search-baseline-v5/`、`citation-rag-v15/`、`citation-rag-v16/`、`citation-rag-v18/`

当前 HEAD 的 38 项单元测试全部通过。所有评测数值来自已提交的逐题 JSON 与 `summary.json`，未手工修改分数。
