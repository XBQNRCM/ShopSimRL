# 来源与复用说明

## 代码与环境

`ShopSimulator/` 现由主仓库以普通目录跟踪。迁移前嵌套仓库的 commit 为 `9bd2fef9b1c06569af32c5012a6eb132d0b239c7`，来源 remote 为 `https://github.com/ShopAgent-Team/ShopSimulator.git`。本地修改包括清洗数据、persona 划分、奖励/observation 契约与回放界面，不能视为上游项目未经修改的发行版。

目录迁移不删除环境源码或历史实验。原嵌套 Git 元数据在执行迁移的本机保存于被忽略的 `tmp/shopsimulator-git-backup/`，不随仓库发布，也不作为新用户运行依赖。训练框架位于 `slime/`，其自带许可文件保留。

根仓库没有另行指定新许可证。本次文档工作不替用户决定原始研究代码的许可，也不替上游代码、模型或数据授予额外复用权利；适用声明沿用各来源自身的文件与条款。未填写虚构作者、论文出版信息或 DOI。

## 初始技能 S₀

公开副本为 `artifacts/cold-start/selected_skillbank.json`，10 个 chunks。它恢复自历史正贡献 run 中 422 条一致的冻结技能上下文，文本与 draft 核对一致。恢复脚本不读取 test reward/success 进行选择，manifest 标记：

```json
{"uses_test_outcomes": false, "validation_refitted": false}
```

本机缺少满足当前严格 identity 检查的那次历史 bare validation，因此恢复产物不是一次重新完成的 Gate A；不更改旧 manifest 来绕过配对检查。来源与 hash 见 [recovery_manifest.json](../artifacts/cold-start/recovery_manifest.json)，实现见 [恢复脚本](../scripts/archive/recover_paired_skillbank.py)。

## 训练与终局技能

四轮 online gates 的 `contributions.json` 保留 400 对 observations、mask、原始 reward、bare reward、delta 和估计元数据。本次报告可用这些已公开数值独立重拟合系数。最终 Sₜ 为 `artifacts/round-003/gate/selected_skillbank.json`，8 个 chunks。

S₀、每轮技能与 curriculum 都保留原始文件和 hash。历史产物中的 Windows/Linux 绝对路径是来源记录，不是发布网页需要访问的服务地址。不要为了让文档看起来整洁而改写冻结 provenance。

## 数据公开边界

`artifacts/analysis/` 新增逐步/组级数值摘录、四条件逐任务 reward/success、可复算区间及图表，不加入原始 persona、完整对话或 privileged analyst transcripts。已有训练/评测发布摘录保留原样。`runs/`、生成 `data/`、W&B、本机日志和 `.env` 不进入主仓库。

统计重采样仅使用冻结任务的数值，不产生新训练或测试调用。分析日期与实验日期不同，不应将文档生成日期当作新实验完成时间。
