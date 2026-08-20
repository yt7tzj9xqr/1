# MiniMax-M3：10 题 baseline 与 Citation-RAG 实验结果

## 实验设置

- 数据：固定的 ReportBench 10 题子集，baseline 与 RAG 使用完全相同的题目。
- 模型：MiniMax-M3，同时用于报告生成与 statement evaluation。
- RAG：以主题检索结果为起点，沿引用关系扩展三层；检索采用无需 API Key 的 OpenAlex、Semantic Scholar、Crossref 级联，并使用本地 SQLite 缓存。
- 评测：沿用 ReportBench Page 7 Table 1 的 reference、cited statement、non-cited statement 三类指标；statement judge 每项执行 3 次并多数投票。
- 本实验未使用 Firecrawl、SerpAPI 或付费 OpenAlex Key。

## 主要结果（10 题宏平均）

| 系统 | Reference precision | Reference recall | Cited match rate | Non-cited factual accuracy | 平均引用数 | 平均 cited 数 | 平均 non-cited 数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniMax-M3 baseline | 0.2226 | 0.01008 | 0.5577 | 0.3320 | 7.7 | 54.4 | 8.5 |
| MiniMax-M3 + Citation-RAG | 0.1142 | 0.00946 | 0.8039 | 0.5809 | 10.5 | 32.5 | 4.1 |
| 变化 | -0.1084 | -0.00062 | +0.2461 | +0.2489 | +2.8 | -21.9 | -4.4 |

## 补充统计（按陈述数量加权）

| 系统 | Supported cited / cited | Cited micro accuracy | Correct non-cited / non-cited | Non-cited micro accuracy |
|---|---:|---:|---:|---:|
| MiniMax-M3 baseline | 263 / 544 | 0.4835 | 37 / 85 | 0.4353 |
| MiniMax-M3 + Citation-RAG | 253 / 325 | 0.7785 | 35 / 41 | 0.8537 |

RAG 有 2 题未抽取出 non-cited statement，baseline 有 1 题未抽取出。主表忠实保留当前评测器的任务宏平均口径，其中无样本任务记为 0；为避免误读，若只对 non-cited 数量大于 0 的任务取宏平均，baseline 为 0.3689（9 题），RAG 为 0.8299（7 题）。

## 结论与下一步

当前 Citation-RAG 对 statement quality 的改进明显：cited match rate 和 non-cited factual accuracy 在宏平均、按陈述加权两种口径下都提高。但是 reference precision 和 recall 没有提高，precision 反而明显下降，说明三层扩展引入了更多“领域相关但不属于 ground truth 引用集合”的论文。因此当前结果可以支持“提升陈述可支撑性和事实准确性”的贡献点，但不能声称三个指标均提高。

在扩展到 30 题及 MiniMax-M2.7 前，应增加 citation graph 的主题约束和重排：每层节点必须保持与根主题/anchor paper 的覆盖度，优先保留同时具有语义相关性、引用路径质量和高引用量的论文，并对最终 evidence top-k 做消融。改进后先重跑同一 10 题；只有 reference precision/recall 不再退化，再冻结配置并扩展到 30 题和 M2.7 全实验。

## v12 优化实验（2026-08-20）

v12 实现了上述改进：技术主题与应用领域组合查询、摘要稀疏时的免费来源补充、父节点覆盖优先的三层图扩展、深度 2/3 节点仅作遍历桥梁、8 篇高置信证据重排、低影响 direct seed 软惩罚，以及单 URL 原子陈述和 References 去重。10 题均成功生成并完成三票评测。

| 指标 | 旧 Citation-RAG | v12 | 变化 |
|---|---:|---:|---:|
| Reference precision（任务宏平均） | 0.1142 | 0.2743 | +0.1601 |
| Reference recall（任务宏平均） | 0.009463 | 0.009405 | -0.000058 |
| Reference micro precision | 0.1048 | 0.2500 | +0.1452 |
| Reference micro recall | 0.005714 | 0.008831 | +0.003117 |
| Ground-truth reference matches | 11 / 105 | 17 / 68 | +6 个命中 |
| Cited match rate（任务宏平均） | 0.8039 | 0.8825 | +0.0787 |
| Cited micro accuracy | 0.7785 | 0.9053 | +0.1268 |
| Non-cited micro accuracy | 0.8537（35/41） | 1.0000（3/3） | +0.1463 |

v12 的 reference precision、总命中数、micro recall、cited 和实际有样本的 non-cited accuracy 都提高。任务宏平均 reference recall 因不同题目的 ground-truth 集合大小差异而微降；因此不能声称该口径也提高，但总命中从 11 增至 17。v12 有 9 题没有生成可评估的 non-cited factual statement，原聚合口径把无样本任务记为 0，产生 0.100 的表面宏平均；这不代表这些陈述判断错误。应同时报告 3/3 的 micro accuracy、样本总数和非空任务数。

下一步应以 v12 作为新候选主配置，在 30 题上验证这些提升能否保持；同时针对宏 recall 为 0 的题目增加 query diversification，而不扩大最终 8 篇 evidence budget，以免重新损害 precision。

## 结果位置

- Baseline 汇总：`artifacts/MiniMax-M3/baseline/full-10-final/summary.json`
- RAG 汇总：`artifacts/MiniMax-M3/citation-rag/full-10-final/summary.json`
- v12 汇总：`artifacts/MiniMax-M3/citation-rag-v12/full-10-final/summary.json`
- 每题结果：对应目录中的 `<arxiv_id>.json` 和 `summary.csv`

`artifacts/`、`runs/`、`cache/` 默认不提交 Git，以避免提交大体积模型输出、缓存以及潜在敏感运行数据；本报告记录了可用于论文撰写的汇总结果。
