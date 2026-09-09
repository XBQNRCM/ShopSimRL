# 实验结果、技能演化与解释

最强的结果是训练后模型在**无外部技能**条件下的表现。这区分最终模型能力与技能的推理时附加作用，但没有分离训练各机制的因果贡献。

## 四个配对 test 条件

| 条件 | Strict reward | 成功率 | 平均模型步数 | 覆盖 |
|---|---:|---:|---:|---:|
| Base，无技能 | 0.5422 | 51.75% | 6.2125 | 400/400 |
| Base + S₀ | 0.5663 | 54.50% | 6.6125 | 400/400 |
| Iter80，无技能 | 0.8501 | 84.25% | 5.7525 | 400/400 |
| Iter80 + Sₜ | 0.8671 | 86.25% | 6.0100 | 400/400 |

![四组测试结果](../figures/test_results.svg)

| 配对成功率差值 | 百分点 | 95% 区间 |
|---|---:|---:|
| Iter80 free − Base free | +32.50 | [+27.50, +37.50] |
| Base + S₀ − Base free | +2.75 | [−1.50, +7.25] |
| Iter80 + Sₜ − Iter80 free | +2.00 | [−0.50, +4.75] |

区间按任务配对做 10,000 次 percentile bootstrap，描述本次训练条件下的测试任务不确定性，不代表跨训练种子变异。两个仅加技能的区间均跨零，不宣称最终技能收益已获统计确证。

Bare Base → Iter80 有 139 个任务从失败变成功、九个从成功变失败；198 个均成功、54 个均失败。Iter80 加 Sₜ 修复 19 个任务、退化 11 个，净增八个成功。总体改善不意味着每个任务改善。

## 验证与技能库变化

| Checkpoint | Bare strict | Random-mask strict | Bare success | 选中 chunks | 候选 |
|---|---:|---:|---:|---:|---:|
| Iter20 | 0.765952 | 0.791661 | 74.75% | 7 | 6 |
| Iter40 | 0.813119 | 0.840690 | 80.00% | 5 | 6 |
| Iter60 | 0.860119 | 0.879774 | 84.75% | 8 | 6 |
| Iter80 | 0.869417 | 0.876024 | 86.00% | 8 | 4 |

![验证与测试配对效果](../figures/validation_and_effects.svg)

随机 gate 评估 current 与候选版本池，而非选中后的完整技能，其均值不能替代 full-skill evaluation。技能库由初始十块收缩至七块、五块，再扩至八块，是淘汰与新增/改写共同作用的结果。

| 轮次 | All-wrong groups | 失败重试 / analyzed cards | Eligible cards | 候选 |
|---|---:|---:|---:|---:|
| R0 | 104 | 87 | 61 | 6 |
| R1 | 57 | 56 | 22 | 6 |
| R2 | 54 | 53 | 20 | 6 |
| R3 | 39 | 39 | 11 | 4 |

![技能演化与失败漏斗](../figures/skill_evolution.svg)

All-wrong 数减 analyzed cards 数并不等于完整技能修复数，因为重试也会发生技术故障。实际 internalization-deficit 数为 9、1、1、0。技术故障、全错策略组、剩余诊断失败、符合提案条件的 cards 是不同阶段。

![贡献历史](../figures/chunk_contributions.svg)

热图按 logical chunk identity 对齐，圆点表示保留的 winner，空白表示本轮未评估；改写保留 logical identity。失败版本保留在可下载系数表。首页[技能浏览器](../index.html#skills)可展开每版 selected bank 的原始中文正文。

## 附加指导也有成本

![奖励分量](../figures/reward_components.svg)

加入 Sₜ 后并非所有分量改善。Iter80 + Sₜ 共 2,404 模型步、276 协议错误；Iter80 free 为 2,301 步、144 错误。总错误 / 总步数为 11.48% 与 6.26%，不同于逐任务错误比例再平均。API 平均每步生成分别为 191.02 与 172.93 token。Full skill 也带来更多 prompt token 和观测时延，但时延取决于硬件与服务条件。

## 排序附录及缺失对照

Base 模型固定 k=4 的额外比较中，top4 成功率为 51.75%，bottom4 为 55.00%，一次 random4 为 54.25%。只完成了 random seed 20260907，其余四个计划随机组没有运行。排序来自另一个本地 Base gate，不能与历史恢复 S₀ 系数混用。结果未确立 held-out ranking 有效性，也不能证明反向排序更好。

本次没有 plain GRPO、固定 S₀、无 mask、无 online gate、q sweep 或多训练种子对照。结果展示该训练系统下模型相对初始 checkpoint 的明显提升，不能确立共同演化优于普通 RL，也不能确定收益由哪个组件产生。验证集复用、主效应系数噪声、chunk 交互及历史 S₀ 来源仍是实质限制。

继续阅读[行为变化](behavior.html)、[优化诊断](optimization.html)或[完整原始实验报告](../reference/docs/experiment_report.html)。
