# MiniMax-M3：30 题完整 Baseline 与 Citation-RAG v12 实验结果

## 实验结论

固定的 ReportBench 30 题子集已全部跑通，MiniMax-M3 baseline 与优化后的 Citation-RAG v12 均为 30/30 有效结果。使用同一 MiniMax-M3 裁判、每条 statement 三票多数决、每批 8 条、每题最多 20 条 non-cited statement。v12 的核心收益是显著提高引用对陈述的支撑质量：cited 宏平均由 46.74% 提升到 80.20%，微平均由 50.53% 提升到 85.77%，逐题 24 胜、2 平、4 负。

## 最终汇总

| 指标 | Baseline | Citation-RAG v12 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Reference precision（宏平均） | 15.38% | 17.33% | +1.95 pp | +12.67% |
| Reference recall（宏平均） | 0.655% | 0.631% | -0.025 pp | -3.76% |
| Reference precision（微平均） | 15.59% | 19.08% | +3.48 pp | +22.35% |
| Reference recall（微平均） | 0.554% | 0.631% | +0.076 pp | +13.79% |
| Cited statement（宏平均） | 46.74% | 80.20% | +33.47 pp | +71.61% |
| Cited statement（微平均） | 50.53% | 85.77% | +35.23 pp | +69.74% |
| Non-cited（全部任务宏平均） | 36.29% | 39.26% | +2.97 pp | +8.18% |
| Non-cited（微平均） | 38.19% | 90.34% | +52.16 pp | +136.58% |

Reference 命中数由 29 增至 33；生成引用总数由 186 降至 173，因此微平均 precision 提升。两组面对同一 5,230 条 gold references。宏平均 recall 略降，但按全部引用计数的微平均 recall 上升，说明新增命中集中在部分任务。

## Statement 数量与覆盖率

| 项目 | Baseline | Citation-RAG v12 |
|---|---:|---:|
| 平均 cited statements / 题 | 40.77 | 18.27 |
| cited statements 总数 | 1,223 | 548 |
| 平均 non-cited statements / 题 | 10.30 | 4.83 |
| non-cited statements 总数 | 309 | 145 |
| non-cited 非空任务 | 28/30 | 14/30 |
| 非空任务平均 non-cited accuracy | 38.88% | 84.13% |

v12 同时减少了约 55.2% 的 cited statements 和约 53.1% 的 non-cited statements。其高分并非只来自“增加引用数量”，而是来自证据筛选、单句单引用和删除无法稳定归因的多来源合成句。不过 non-cited 指标存在明显覆盖率差异：v12 只有 14 道题有可评 non-cited statement，所以 90.34% 的微平均只能解释为“被抽取出的 145 条未引用事实中有多少得到证据支持”，不能解释为全部报告事实的总体正确率。

## 逐题配对胜负（v12 相对 baseline）

| 指标 | 胜 | 平 | 负 |
|---|---:|---:|---:|
| Reference precision | 10 | 14 | 6 |
| Reference recall | 5 | 22 | 3 |
| Cited statement | 24 | 2 | 4 |
| Non-cited 原始任务分数 | 11 | 6 | 13 |

Non-cited 的逐题胜负受空集合记 0 的规则影响；两组同时存在 non-cited statement 的任务仅 13 道，因此该行不应单独作为主要结论。毕业论文最稳妥的主贡献表述是：Citation-RAG v12 在相同模型、相同题目和相同裁判协议下，显著且广泛地提高了 cited statement 的支撑准确率，同时提高 reference 微平均 precision/recall，并降低报告中的无引用事实数量。

## 优化版实现

检索不使用 Firecrawl、SerpAPI 或付费 OpenAlex key。系统依次使用免费 OpenAlex、Semantic Scholar、Crossref，并用本地 SQLite 缓存搜索与模型响应；从主题种子构建最多三层、40 篇论文的引文图。写作阶段只保留深度 0/1 的高相关证据，结合主题相关性、引用影响力、父节点覆盖和重复惩罚排序；向 M3 提供最多 8 篇核心证据。生成后执行引用规范化与清洗，每个可验证引用句只保留一个 URL，删除 bibliography 和同句多 URL 合成，降低“引用存在但不能完全支撑整句”的风险。

## 稳定性测试与修复

- 两个系统均生成 30/30 有效 `result.json`；旧 `error.json` 是成功重试前的历史记录，不是最终失败。
- RAG 长报告的生成预算提升到 65,536 tokens，解决 `finish_reason=length` 空输出。
- statement 裁判对返回数量不符、非法/截断 JSON 和长度失败自动二分批次。
- non-cited 抽取先复用 v3 缓存，发生长度失败时切换到 v4 分批递归抽取。
- MiniMax 客户端对 429/5xx、超时、连接重置、远端断开和 `IncompleteRead` 最多退避重试 8 次。
- 14 项自动化回归测试全部通过。

## 可复查产物

- Baseline 完整汇总：`artifacts/MiniMax-M3/baseline/full-30-final/summary.json`
- Baseline 逐题表：`artifacts/MiniMax-M3/baseline/full-30-final/summary.csv`
- v12 完整汇总：`artifacts/MiniMax-M3/citation-rag-v12/full-30-final/summary.json`
- v12 逐题表：`artifacts/MiniMax-M3/citation-rag-v12/full-30-final/summary.csv`
- 生成结果：`runs/MiniMax-M3/baseline/` 与 `runs/MiniMax-M3/citation-rag-v12/`

这些运行产物默认不提交 Git，以避免仓库被大体积报告、裁判明细和缓存污染；代码、测试与本报告提交 Git。最终 commit 之前的关键稳定性修复包括 `f77669e`、`ec2e519`、`eca8b20`、`8953861` 和 `80701ae`。
