# 文档导航

项目名称为 **IntraSkill**，代码仓库为 **ShopSimRL**。当前发布结果对应 Qwen3.5-4B、四轮训练、Iter80 checkpoint。对外摘要见 [English README](../README.md) / [中文 README](../README.zh-CN.md)。

| 阅读目标 | 文档 |
|---|---|
| 了解实际实验结果、数据口径和局限 | [实验报告](experiment_report.md) |
| 挖掘生成长度、成功失败步数、搜索、商品访问与协议变化 | [行为分析](behavior_analysis.md) |
| 检查 W&B 优化器更新、熵、KL、梯度与吞吐 | [优化与系统分析](optimization_analysis.md) |
| 复算分析、理解 CSV/JSON 和图表 | [分析产物说明](../artifacts/analysis/README.md) |
| 运行或发布中英双语项目页 | [项目页](project_page.md) |
| 理解方法的研究动机 | [研究方法草案](proposal.md) |
| 理解当前训练实现和配置 | [训练指南](training.md) |
| 理解 Trace2Skill 候选生成 | [Trace2Skill 方案](trace2skill.md)、[冷启动实现](trace2skill_cold_start.md) |
| 核验 gate 的配对与估计语义 | [Paired validation](paired_validation.md) |
| 核验生成/训练样本的统计口径 | [动态过滤](dynamic_sampling_filter.md)、[监控指标](training_monitoring.md) |
| 运行评测、理解 trace schema | [Runtime 与评测](runtime_eval.md) |
| 启动环境与本地轨迹回放 | [环境指南](../ShopSimulator/README.md)、[环境契约](../ShopSimulator/docs/ENVIRONMENT.md)、[Replay UI](replay_ui.md) |
| 了解来源、历史恢复和复用边界 | [来源说明](provenance.md) |
| 查阅机器配置和历史实验 | [Archive](archive/README.md) |

## 文档与实验事实的优先级

实验数字以冻结 `artifacts/` 和可复算的 `artifacts/analysis/` 为准；实现契约以当前代码、测试及相应工程文档为准。`proposal.md` 是研究草案，其中自适应课程等可能方案并不代表本次运行已经实现或验证。

本次 `q` 全程固定 0.20。对外展示和数据下载只包含四轮完整训练、四轮 gate 与 Iter80 test，训练范围为 step 1–80。
