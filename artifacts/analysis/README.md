# 可复算分析 / Reproducible analysis

这些文件由 `scripts/analyze_training.py` 和 `scripts/plot_training.py` 生成。主实验范围为 rollout steps 1–80，四轮 gate，最终 checkpoint 为 Iter80。无需 `runs/` 或模型 API 即可从这里的数值摘录重建分析与图表。

```bash
python -m pip install -e ".[analysis]"
python scripts/analyze_training.py
python scripts/analyze_behavior.py
python scripts/analyze_wandb.py
python scripts/plot_training.py
python scripts/plot_behavior.py
python scripts/plot_wandb.py
```

只有重新提取本地完整实验时才使用 `--refresh-local`。该选项读取 W&B 文本日志、每个 group 的 triage，以及四组 test traces，不修改历史产物。

| 文件 | 内容与单位 |
|---|---|
| `analysis.json` | 项目页和图表的统一数据，包括 test、配对区间、round、step、贡献和五个技能版本 |
| `training_steps.csv` | 每行一个 rollout step，严格覆盖四轮 1–80；空值代表未记录，不是零 |
| `training_sources.json` | 每个 step 的原始日志相对路径；相同 step 的重复来源必须数值一致 |
| `training_groups.csv` | 每行一个生成 group；分类、treatment、可评分数、奖励与 retry 是否存在；不等同于 trainer-admitted groups |
| `test_task_outcomes.json` | 400 个共同 test task 的四条件 reward/success；不含对话、persona 或商品答案 |
| `local_extraction_audit.json` | traces hash、尝试数、唯一完成数与 summary 复算结果 |
| `rounds.csv` | 四轮输入/输出技能大小、失败/候选数量与系数复算误差 |
| `chunk_contributions.csv` | 每轮每个 intervention 的有符号系数、logical ID、operation 和 selection status，包含 rewrite losers |
| `provenance.json` | 输入 JSON/CSV 统一 CRLF→LF 后的 SHA-256 和数值依赖版本，兼容历史 Git 文件的不同换行 |
| `behavior/` | 20,480 训练 / 3,600 评测 episode，逐 turn 数值 gzip CSV、分层统计、配对区间与来源 |
| `wandb/` | 四个 finished run 的数值 history、80 rollout steps、160 optimizer updates、来源哈希 |
| `figures/` | 13 组独立 SVG 与 PNG；英文坐标与图例，中英说明在网页 |

## 核验口径

- 从完整 traces 每个任务取唯一 completed 记录，拒绝重复 completed；失败尝试不是新的任务样本。
- 每个最终条件必须覆盖同一组 400 个任务，提取时重新调用项目 summary 实现，校验主 reward 与 success。
- 四轮 gate 从实际 `mask` 与逐任务 `reward-bare_reward` 重拟合无截距 OLS，检查列秩和系数一致性。
- Bootstrap 联合重采样任务的两条件结果，10,000 次，使用比较脚本相同的 Python `random.Random` 和按指标哈希派生的种子。新统一复算的跨 checkpoint 区间可能与未记录完整随机数约定的历史 cross-checkpoint 摘录有轻微差别；历史文件不被覆盖。
- task-level CI 不包含训练随机种子的不确定性。系数正负用于当前筛选规则，不是统计显著性标签。

## 图表

| 名称 | 解释 |
|---|---|
| `test_results` | 四条件 test strict reward 与成功率 |
| `training` | 全量生成奖励、实际 skill-free 比例、all-wrong 比例、zero-std 丢弃率 |
| `validation_and_effects` | 四轮 bare/masked val 曲线，以及 test 配对效果区间 |
| `skill_evolution` | active chunk 数与失败分析漏斗 |
| `chunk_contributions` | 每个逻辑技能的胜出版本系数；点表示保留，灰色表示未评估 |
| `reward_components` | Base free、Iter80 free、Iter80 + Sₜ 的分量比较 |

曲线展示原始点与 5-step trailing mean。生成样本与进入 trainer 的样本保持不同分母；完整解释见 [实验报告](../../docs/experiment_report.md)。

新增 `behavior_outcomes`、`behavior_step_distribution`、`behavior_fixed_validation`、`behavior_paired_effects`、`behavior_phases` 五组行为图，以及 `optimization`、`systems` 两组 W&B 图。行为曲线除明确标注外不平滑。训练 token 来源是保存输出的重编码，评测 token 来源是 API usage；W&B response length 是已接纳 Sample 的长度，三个口径不能混用。详见[行为报告](../../docs/behavior_analysis.md)与[优化报告](../../docs/optimization_analysis.md)。

默认复算不联网。原始行为刷新需 `tokenizers` 与官方 Qwen tokenizer JSON，执行 `python scripts/analyze_behavior.py --refresh-local --tokenizer /path/to/tokenizer.json`。W&B 刷新需可选 `[wandb]` 依赖和读取权限，执行 `python scripts/fetch_wandb.py`；凭据仅在请求时读取，不导出。

W&B manifest、behavior gzip provenance 和站点 build manifest 使用原始字节 SHA-256。`.gitattributes` 对 `artifacts/analysis/**` 禁用 Git 换行转换，以保存冻结数值导出和其字节哈希。只有主 `provenance.json` 的历史 JSON/CSV 输入先统一换行；其 normalization 字段明确记录这一规则。
