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

## 结果位置

- Baseline 汇总：`artifacts/MiniMax-M3/baseline/full-10-final/summary.json`
- RAG 汇总：`artifacts/MiniMax-M3/citation-rag/full-10-final/summary.json`
- 每题结果：对应目录中的 `<arxiv_id>.json` 和 `summary.csv`

`artifacts/`、`runs/`、`cache/` 默认不提交 Git，以避免提交大体积模型输出、缓存以及潜在敏感运行数据；本报告记录了可用于论文撰写的汇总结果。
