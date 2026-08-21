# MiniMax-M3 固定 10 题实验记录（2026-08-21）

## 结论

固定 10 题上，`search-baseline-v4` 和 `citation-rag-v15` 均已完成生成、reference、cited statement、non-cited statement 全流程评测。与 baseline 相比，RAG 的 reference precision 从 0.2221 提升到 0.2957，cited match 从 0.5177 提升到 0.9397；reference recall 从 0.01704 降到 0.01339。Non-cited 的 42 条有效语句 micro accuracy 为 0.6429，高于 baseline 的 0.4146，但 RAG 有 4 题没有抽取到 non-cited factual statement，若沿用聚合器把空任务记为 0 的口径，宏平均只有 0.2692。因此当前可以支持“RAG 明显提高引用精度和引用一致性、减少无根据陈述”的结论，尚不能支持“三项指标全部提高”的结论。

`citation-rag-v15` 中前 8 题与后 2 题跨越了一次搜索策略代码变更，是优化阶段 pilot，而不是最终冻结实验。最终代码已恢复五查询高召回路径并加入证据修复，必须使用新系统名完整重跑同一 10 题后，才能作为毕业论文最终主表。

## 与论文 Table 1 对照

下表均为任务宏平均；论文原表使用 100 题和不同评判模型，本实验是固定的偏难 10 题且使用 M3 代理评判，绝对数值只能参照，baseline/RAG 的同配置配对比较更可靠。

| 系统 | Ref. precision | Ref. recall | Ref. count | Cited match | Cited count | Non-cited accuracy | Non-cited count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 论文 Gemini 2.5 Flash | 0.237 | 0.012 | 5.47 | 0.4488 | 12.10 | 0.9852 | 11.50 |
| 论文 Gemini 2.5 Pro | 0.269 | 0.010 | 4.27 | 0.5924 | 6.58 | 0.9608 | 9.35 |
| 论文 o3 | 0.299 | 0.031 | 12.26 | 0.3143 | 16.16 | 0.8222 | 11.51 |
| 论文 Claude 4 Sonnet | 0.337 | 0.021 | 6.74 | 0.7367 | 14.93 | 0.9264 | 17.07 |
| M3 search-baseline-v4 | 0.2221 | 0.01704 | 13.7 | 0.5177 | 58.1 | 0.3987 | 12.3 |
| M3 citation-rag-v15 pilot | 0.2957 | 0.01339 | 10.1 | 0.9397 | 19.9 | 0.2692 | 4.2 |

这 10 题平均每题有 192.5 条 gold reference，reference recall 的分母远大于生成报告实际合理引用数。因此 recall 偏低不等价于检索完全失效，但它仍然揭示了写作阶段没有充分利用已召回 gold 文献的问题。

## 配对变化

| 指标 | Baseline v4 | RAG v15 pilot | 变化 |
|---|---:|---:|---:|
| Reference precision（宏） | 0.2221 | 0.2957 | +0.0736（+33.1%） |
| Reference recall（宏） | 0.01704 | 0.01339 | -0.00365（-21.4%） |
| Reference 命中数 | 30 | 23 | -7 |
| 生成引用数 | 137 | 101 | -36 |
| Cited match（宏） | 0.5177 | 0.9397 | +0.4220 |
| Cited micro accuracy | 288/581 = 0.4957 | 183/199 = 0.9196 | +0.4239 |
| Non-cited raw macro | 0.3987 | 0.2692 | -0.1296 |
| Non-cited micro accuracy | 51/123 = 0.4146 | 27/42 = 0.6429 | +0.2282 |
| Non-cited 非空任务宏平均 | 0.3987（10 题） | 0.4486（6 题） | +0.0499 |

RAG 的高 cited match 不是通过增加大量引用得到的：其 cited statement count 从 58.1 降到 19.9，接近论文普通模型的量级。这说明原子化陈述、每句绑定单一 URL、删除多来源混合句和生成后证据修复的方向有效。Baseline 的 cited count 仍明显过高，是 cited 与 non-cited 指标不稳定的主要原因之一。

## 检索与引用图审计

- 新 baseline 每题保留 16 篇写作证据，10 题候选池共命中 31 篇 gold，最终报告命中 30 篇。
- RAG 的三层图共保存 362 个去重节点，图池命中 36 篇 gold，比 baseline 多 5 篇；最终报告只命中 23 篇。
- 因此当前 RAG 的主要瓶颈已经从“图里完全没有目标文献”转移到“写作证据选择和引用预算丢掉已召回的正确文献”。下一轮不应继续无上限扩大图，而应提高从 pool 到 final citation 的转化率。
- 深度 2/3 节点适合帮助遍历，不应默认写入报告；当前实现只允许 depth 0/1 进入写作证据，并至少保留约 65% 直接搜索证据，避免高被引但偏题的深层节点挤占主题核心文献。

## 当前免费实现

1. M3 规划 5 个学术检索查询，查询并发执行；MiniMax Search 是主要网页搜索入口。
2. OpenAlex 和 Semantic Scholar 只作为免费的结构化元数据补充，用于 DOI、摘要、引用数及引用边；无付费 Key 时允许降级，429 会触发熔断，不能阻塞主流程。
3. 网页正文优先使用本地 `pdftotext` 读取 arXiv PDF，并把正文写入 SQLite 缓存；普通页面读取失败时使用搜索摘要。没有使用 Firecrawl 或 SerpAPI，也没有新增付费 API。
4. Baseline 与 RAG 共用查询预算、候选清洗、年份截止、目标 survey 排除、M3 重排、网页读取和缓存。RAG 的额外部分只包括三层引用图、深度感知证据选择和更严格的 grounded writing，因此不会通过故意削弱 baseline 制造提升。
5. 评测实现复刻 Table 1 的三个字段，但 cited/non-cited 均由 M3 独立投票，是 `M3-only proxy`；论文 non-cited 使用 Gemini Pro/Flash 六票，因此二者不能宣称严格同口径。

## 优化阶段发现的失败配置

`search-baseline-v5` 尝试把五次搜索改成“3 个初始查询 + 阅读结果后生成 2 个反馈查询”。10 题全部完成后，reference precision 为 0.2197、recall 为 0.01421、平均引用 9.3 篇、总命中 16 篇，明显低于 v4 的 30 篇命中；cited 宏平均为 0.7318、micro accuracy 为 0.8798、平均 18.3 条；non-cited micro accuracy 为 0.5085，共评判 59 条、7 个非空任务。结果说明证据修复确实改善 cited consistency，但反馈搜索损害了 reference coverage。原因不是反馈搜索思想本身错误，而是反馈上下文压缩后，M3 倾向生成过窄或重复的题名查询；匿名学术源的补充结果又受速率限制。该配置保留为消融，不作为最终 baseline。当前主路径已经恢复五查询完整召回。

## 下一轮冻结实验标准

最终 10 题必须从同一 Git commit、同一缓存策略、同一配置一次性生成，建议命名为 `search-baseline-v6` 与 `citation-rag-v16`。验收时同时报告：

- reference 宏/微 precision、recall、平均引用数和总命中数；
- cited 宏/微 accuracy 与 statement count；
- non-cited raw macro、micro、非空任务宏平均、非空任务数与 statement count；
- 检索池 gold 命中、最终引用 gold 命中及二者转化率；
- 10/10 成功率、缓存命中情况、单题和总墙钟时间。

只有冻结版本在 10 题上至少保持 reference precision ≥ 0.237、reference recall ≥ 0.012、cited match ≥ 0.75，并且 non-cited 有足够非空样本，才扩展到 30 题和 MiniMax 2.7。Reference recall 不应通过引用无关论文换取，cited match 也不应通过生成极少语句虚增。

## 2026-08-21 混合免费检索与证据修复验证

本轮恢复了 OpenAlex、arXiv、Crossref 和 Semantic Scholar 对中心查询的免费结构化补充，并将匿名请求限制为 8 秒、失败快速降级；MiniMax Search 仍承担五个规划查询。10 题候选池审计从纯 MiniMax 网页搜索的 15--17 个 gold 命中提高到 31 个，且恢复了引用图所需的结构化 work ID。该实现不要求 OpenAlex API Key，也不使用 Firecrawl 或 SerpAPI。

| 系统 | Ref. P | Ref. R | Ref. count | Cited match | Cited count | Non-cited raw macro | Non-cited micro | 非空任务 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M3 search-baseline-v5 | 0.2197 | 0.01421 | 9.3 | 0.7318 | 18.3 | 0.4021 | 0.5085 (30/59) | 7/10 |
| M3 citation-rag-v16 | 0.2423 | 0.01941 | 9.7 | 0.8051 | 19.5 | 0.1800 | 0.6000 (27/45) | 3/10 |

RAG 相对本轮 baseline 的 reference precision 提高 10.3%，reference recall 提高 36.6%，gold reference 总命中从 16 提高到 25，cited match 从 0.7318 提高到 0.8051。RAG 的 non-cited raw macro 不能直接解释成“事实准确率只有 18%”：聚合器把 7 个没有抽取到 non-cited statement 的任务记为 0；实际评判的 45 条语句 micro accuracy 为 0.6000，3 个非空任务的宏平均同为 0.6000。

这次运行也定位了最后一个主要问题：证据编辑有时删除过多内容，造成空或近空报告，进而把 cited/non-cited 宏平均记为 0。运行期间代码加入了三道质量门：拒绝低于最低词数的修复结果、拒绝来源覆盖坍缩的结果、对引用清洗后坍缩的报告执行恢复生成；随后又修复了句号后方括号引用被错误合并的问题，当前 HEAD 为 35/35 测试通过。因此 v5/v16 是问题定位和优化验证，不能冒充严格冻结主实验；最终主表必须在当前单一 commit 上使用新系统名重跑同一 10 题，再扩到 30 题。

## 当前冻结 baseline 候选（search-baseline-v10）

`search-baseline-v10` 在统一质量门与方括号引用归位修复后完成 10/10 题，所有报告均超过 600 词且至少包含 8 个独立来源。其 reference precision 为 0.2350、recall 为 0.01298、平均引用 10.8 篇、总命中 25 篇；cited 宏平均为 0.7149、micro accuracy 为 0.7378、平均 28.6 条；non-cited raw macro 为 0.2792、micro accuracy 为 0.6825（43/63），5 个非空任务的宏平均为 0.5583。相对 v4，cited 宏平均从 0.5177 提升到 0.7149，non-cited micro 从 0.4146 提升到 0.6825；reference precision 从 0.2221 提升到 0.2350，但宏 recall 从 0.01704 降到 0.01298。该结果没有短报告或零引用报告造成的有利偏差，可作为当前公平 baseline；仍需用当前同一 commit 重跑 RAG 后才能形成最终配对主表。

## 可复现文件

- Baseline v4：`metrics/MiniMax-M3/search-baseline-v4/reference/` 与 `metrics/MiniMax-M3/search-baseline-v4/full/`
- RAG v15 pilot：`metrics/MiniMax-M3/citation-rag-v15/reference/` 与 `metrics/MiniMax-M3/citation-rag-v15/full/`
- 反馈搜索消融：`metrics/MiniMax-M3/search-baseline-v5/reference/` 与 `metrics/MiniMax-M3/search-baseline-v5/full/`
- 混合免费检索 RAG 验证：`metrics/MiniMax-M3/citation-rag-v16/reference/` 与 `metrics/MiniMax-M3/citation-rag-v16/full/`
- 当前冻结 baseline 候选：`metrics/MiniMax-M3/search-baseline-v10/reference/` 与 `metrics/MiniMax-M3/search-baseline-v10/full/`
- 旧版对照：`metrics/MiniMax-M3/search-baseline-v3/` 与 `metrics/MiniMax-M3/citation-rag-v13/`
- 生成结果和来源池：`runs/MiniMax-M3/<system>/<arxiv_id>/result.json`（默认不提交 Git）
- 检索审计：`scripts/audit_live_retrieval.py`

所有代码修改必须先通过 `PYTHONPATH=src python -m unittest discover -s tests -v`；当前 HEAD 为 35/35 通过。
