# MiniMax-M3 固定 10 题检索修复实验（2026-08-21）

## 结论先行

本轮已完成同一固定 10 题上的 `search-baseline-v3` 与 `citation-rag-v13` 全流程生成和三类指标评估，10/10 题均成功。RAG 的 reference precision 从 0.1690 提升到 0.2200，cited match rate 从 0.4954 提升到 0.7914；但 reference recall 从 0.00863 降到 0.00748，non-cited 的任务宏平均也因 4 个零样本任务而从 0.4769 降到 0.4042。因此，目前只能确认“reference precision 与 cited consistency 提高”，不能声称 Table 1 的全部指标都提高，也不应直接扩展到 30 题。

## 与论文方法的核对

论文中的普通 base model 不是裸模型：模型通过原生工具调用使用 SerpAPI Google Search 和 Firecrawl link reader，每题最多 5 次工具调用。cited evaluation 先抓取被引用网页全文、定位支持段落，再验证陈述一致性；non-cited evaluation 使用联网的 Gemini 2.5 Pro 和 Flash，各判断 3 次，共 6 票。

当前实现的对应关系如下：

- M3 规划 5 个查询并发调用 MiniMax Coding Plan Search，替代 SerpAPI，不需要新增第三方付费 Key。
- 最终候选网页由本地 `WebPageReader` 并发读取并写入 SQLite 缓存，替代 Firecrawl；失败时退回搜索摘要。
- baseline 和 RAG 共享相同搜索、年份过滤、禁止目标 survey、网页读取和缓存，避免人为压低 baseline。
- cited evaluation 使用网页正文或摘要，由 M3 独立投票 3 次；non-cited 对每条 claim 重新联网检索，再由 M3 投票 3 次。
- 这仍是“近似复刻”：当前不是模型原生 function-call 循环，non-cited 也不是论文的双 Gemini 六票，因此绝对值只能谨慎对照，baseline/RAG 的同配置配对比较更可信。

## Table 1 对照（任务宏平均）

| 系统 | Ref. precision | Ref. recall | Ref. count | Cited match | Cited count | Non-cited accuracy | Non-cited count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 论文 Gemini 2.5 Flash | 0.237 | 0.012 | 5.47 | 0.4488 | 12.10 | 0.9852 | 11.50 |
| 论文 Gemini 2.5 Pro | 0.269 | 0.010 | 4.27 | 0.5924 | 6.58 | 0.9608 | 9.35 |
| 论文 o3 | 0.299 | 0.031 | 12.26 | 0.3143 | 16.16 | 0.8222 | 11.51 |
| 论文 Claude 3.7 Sonnet | 0.337 | 0.021 | 6.74 | 0.7367 | 14.93 | 0.9264 | 17.07 |
| M3 search-baseline-v3（本轮 10 题） | 0.1690 | 0.00863 | 7.70 | 0.4954 | 28.70 | 0.4769 | 10.80 |
| M3 citation-rag-v13（本轮 10 题） | 0.2200 | 0.00748 | 4.80 | 0.7914 | 20.80 | 0.4042 | 3.30 |

本轮固定 10 题并非论文 100 题的等价缩小样本：它们平均有 192.5 条 gold reference、中位数 260；完整 100 题平均 162.0、中位数 143。因此该 10 题在 reference recall 上明显偏难，不能把与 Table 1 的差值全部归因于模型或检索器。

## 配对结果与加权指标

| 指标 | Baseline | RAG v13 | 变化 |
|---|---:|---:|---:|
| Reference precision（宏） | 0.1690 | 0.2200 | +0.0510（+30.2%） |
| Reference recall（宏） | 0.00863 | 0.00748 | -0.00115 |
| Reference matches / cited refs | 13 / 77 | 10 / 48 | precision 提高、覆盖下降 |
| Cited match（宏） | 0.4954 | 0.7914 | +0.2961 |
| Cited micro accuracy | 132/287 = 0.4599 | 173/208 = 0.8317 | +0.3718 |
| Non-cited macro accuracy | 0.4769 | 0.4042 | -0.0728 |
| Non-cited micro accuracy | 45/108 = 0.4167 | 18/33 = 0.5455 | +0.1288 |
| Non-cited 非空任务宏平均 | 0.4769（10 题） | 0.6736（6 题） | +0.1967 |

RAG 的 non-cited 宏平均较低主要是聚合器把 4 个“未抽取出 non-cited factual statement”的任务记为 0；这不等价于 4 题全部事实错误。因此论文撰写时必须同时给出 count、micro accuracy 和非空任务数，不能只选择有利口径。

## 为什么仍低于论文 baseline

1. **搜索页 recall 仍是首要瓶颈。** 稳定配置每题保留 12 篇时共命中约 15–16 个 gold；把候选池扩大到最多 30 篇后仍只有 16 个，说明不是 top-k 截断，而是搜索结果本身没有返回目标文献。
2. **一次性查询规划不等于论文的 agentic tool loop。** 当前 M3 在看见搜索结果之前就同时生成 5 个查询；论文模型可以阅读结果后决定下一次搜索。landmark-only 消融只有 13/120 命中，证明单纯让 M3 猜经典论文题名不稳定。
3. **三层引用图当前实际没有完整展开。** MiniMax Search 返回 URL、标题和 snippet，但没有 `referenced_work_ids`；匿名 OpenAlex/Semantic Scholar 又持续 429。因此 RAG 结果平均仅保存 6.0 篇，主要是“搜索池 + 网页正文 RAG”，不能把本轮描述成已经完成三层 citation graph。
4. **网页正文覆盖不完整，但不是唯一主因。** baseline 平均 11.7 篇候选中 5.8 篇抓到正文，RAG 平均 6.0 篇中 3.5 篇抓到正文。baseline 在正文证据上的 cited 支持率为 0.429、摘要回退为 0.473；两者都低，说明生成器的复合句和证据外推比“是否抓到正文”更关键。RAG 对应两类证据分别为 0.841 和 0.825，原子句约束确实有效。
5. **报告长度不匹配。** baseline 平均 1574 词、28.7 条 cited statements，远高于 Table 1 普通模型的 cited count；RAG 平均 712 词、20.8 条 cited statements，虽明显下降但仍偏多。
6. **目标 survey 泄漏会虚增 Table 1 可比成绩。** 论文 prompt 明示原 survey 标题并要求“不要引用”，但未证明工具层禁止搜索或阅读它。当前实现会从检索池移除目标 survey，属于更严格的 clean setting。若直接读取目标 survey 的 bibliography，reference 分数会接近对 gold 的泄漏，不应作为贡献点；建议只做单独的 leakage ablation。

## 运行效率

- baseline：2 个任务并发，10 题墙钟约 574 秒（9.6 分钟），单题平均 96.8 秒，最长 256.4 秒。
- RAG：2 个任务并发，首次运行含 MiniMax 重试及一题恢复，10 题墙钟约 1718 秒（28.6 分钟），单题结果记录平均 191.6 秒。
- 已把 RAG 最大输出从 65,536 降到 24,576 token；中断后单独恢复的流形题约 116 秒完成。
- 推荐并发层级：任务/模型 2 路、每题搜索 5 路、网页读取 4 路；公共学术 API 保持各自串行限速，不能把所有网络请求无差别并发。

## 下一步实现顺序

1. **先实现论文式反馈搜索 baseline。** 将 5 次搜索预算改为“3 个初始查询 → M3 阅读标题/snippet → 2 个精确补充查询”，并记录每次工具轨迹；固定相同 10 题，只在检索候选 gold 命中显著超过 16 后才重生成。
2. **为 RAG 增加真正的 citation-edge resolver。** 从非目标 seed 网页提取 DOI、arXiv ID、`citation_reference` 和参考文献区；解析得到的题名再通过 MiniMax Search 定位原文。目标 survey 只用于独立 leakage ablation，主实验必须排除。
3. **增加两阶段证据重排。** 第一阶段按主题、查询覆盖和来源质量召回；第二阶段让 M3 在不看 gold 的前提下选 5–6 篇最可能支持任务核心分类的原始论文。扩大搜索池但不扩大最终写作 evidence。
4. **继续缩短报告并做 claim-level gating。** baseline 目标 700–900 词，RAG 600–800 词；每句生成前绑定一个 evidence span，生成后删除无法定位支持片段的句子。
5. **修正评估可比性。** cited 增加“最相关段落定位”步骤；non-cited 若只有 M3，则使用 6 次独立检索投票并标记为 `M3-only proxy`，不要与论文双 Gemini 分数写成严格同一实验。
6. **停止扩展到 30 题。** 只有固定 10 题同时满足 reference precision 不低于 0.237、reference recall 不低于 0.012、cited 不低于 0.75，且 non-cited micro accuracy 在足够样本上提高后，再冻结配置运行剩余 20 题和 M2.7。

## 可复现位置

- baseline 汇总：`metrics/MiniMax-M3/search-baseline-v3/full/summary.json`
- RAG 汇总：`metrics/MiniMax-M3/citation-rag-v13/full/summary.json`
- 每题 reference：对应系统的 `reference/<arxiv_id>.json`
- 每题 statement 细节：对应系统的 `full/<arxiv_id>.json`
- 生成报告和来源池：`runs/MiniMax-M3/<system>/<arxiv_id>/result.json`（默认不提交 Git）
- 检索审计工具：`scripts/audit_live_retrieval.py`

`runs/`、`cache/` 和 `artifacts/` 默认忽略，以避免提交大体积模型输出和缓存；本轮 `metrics/` 与该报告提交到 GitHub。
