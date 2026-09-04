# 统一的 paired-delta random masked validation

Cold-start Gate A 与训练阶段的 online gate 都使用同一个估计式：

\[
R_i-B_i=C_i^\top\beta+\epsilon_i,
\qquad
\hat\beta=\arg\min_\beta\sum_i(R_i-B_i-C_i^\top\beta)^2.
\]

- \(B_i\)：当前冻结 checkpoint 在 validation task \(i\) 上**完全不带 skill**的实测 reward。
- \(R_i\)：同一 checkpoint、同一 task 在随机 mask \(C_i\) 下的实测 reward。
- \(C_i\)：实际注入的 chunk/version dummy vector；不中心化、不增加常数列。
- 回归因变量为逐任务差值，不是 \(R_i-\bar B\)；**不拟合截距**。结果中的 `intercept: 0.0` 只是固定值元数据，不是估计的 bare score。

主指标使用 `reward`（当前等于 `r_strict`）；每个完整的 `r_*` 分量分别用自己的 \(R_{i,m}-B_{i,m}\) 做同样回归。全零 mask 也是有效 observation，不能丢弃或把其 delta 强制置零；两次随机 rollout 即使都不注入 skill，也可能有不同 reward。

## 数据复用与运行顺序

1. 冻结当前 checkpoint 及验证配置，用原始 `plan` / `run` evaluator 完成 bare validation；如果已有完全匹配的 run，直接复用，不再采样。其 `summary.json` 就是这一 checkpoint 对外报告的 skill-free validation score。
2. gate 配置中的 `experiment_config` 引用同一份 bare YAML，`bare_run_dir` 指向该 evaluator run（通常为 `experiment.output_dir / experiment.name`）。gate 只运行 masked validation，不在内部另跑 bare。
3. 按 `(split, task_id, sample_id)` 一对一配对。当前协议为完整的固定 `val` split、`repeats: 1`、每个任务一条 bare 和一条 masked rollout；常规 400-task split 对应 400 对 observations，而非 800 条独立回归样本。
4. 冷启动复用初始 checkpoint 的 bare val；训练每次 gate 复用**该次冻结 checkpoint**已经用于报告 bare validation score 的同批 traces。不能复用前一 checkpoint 的 baseline，也不能拿 train group、full-skill retry、test score 或只有 summary 的 run 替代。
5. 训练时 `model.checkpoint_id` 必填，并随实际权重更换（建议使用 checkpoint 路径、revision 或 hash）。它进入 manifest/trace/fingerprint，但不发送给模型 API。相同 endpoint/model name 可能承载不同权重，仅这两项不足以认定同一 checkpoint。该标识由操作者提供，不是代码对服务权重的自动验证；两次评测及失败补测期间都必须保持服务冻结。

训练目前采用有限 round 后暴露 checkpoint、运行通用 evaluator、再运行 online gate 的显式流程；并未把 gate 嵌入每个训练 step。若外部 trainer 已有 bare eval，必须保存兼容的完整 run manifest 与 task traces，不能只传一个聚合分数。操作入口见 [training.md](./training.md)，冷启动见 [trace2skill_cold_start.md](./trace2skill_cold_start.md)。

## 配对校验与恢复

共享核心逻辑在 `shopsimrl/paired_validation.py`，无截距拟合在 `shopsimrl/trace2skill_evaluation.py`，由两条 gate 共用；统计检验/实验比较仍放 `scripts/`。

- 开始模型调用前检查 bare manifest、完整 coverage、终局可评分状态，以及 split/task/sample、checkpoint/model、endpoint、采样参数、基础 prompt、环境配置、runtime/action protocol 一致性；启用 `send_seed` 时还要求 job seeds 一致。未发送 seed 时，seed 相等不构成共享随机数保证。
- bare 必须是 `NoSkills` 且 `selected_skills=[]`；不能用一次碰巧全零的 masked run 充当 bare run。
- 回归前逐条检查实际注入 IDs 与 assignment 一致，两组实际 provenance（包括环境版本）一致，所有 reward 分量完整、有限，且 `reward == r_strict`。未通过校验不产出新的 selection，不静默降级为带截距回归。
- 允许 failed attempt 后补测成功；拒绝同一 episode 多条 completed 记录，避免按完成先后取 latest 引入选择偏差。多个进程不要写同一个 run；重复采样必须单独设计、编号和分析。
- estimator 与 bare 数据摘要 hash 写入 gate 语义计划，基线变化后不能在同一目录 resume。旧带截距 gate 的目录也不能直接续跑新方法；当前 cold-start 配置写入 `gate-a-paired`，旧产物不改写。

此前跨服务的 paired-delta 重分析是基于“分布等价”的探索性假设，不能当作上述严格配对检查已通过。标准 gate 不默认放行后端或解码配置不一致；也不要改历史 manifest 来绕过校验。

## 输出与解释边界

`summary.json` 保留 masked rollout 的原始指标；bare run 的 `summary.json` 保持原样。`contributions.json` 记录 estimator/formula、baseline 来源与 hash、每项的 `masked_mean` / `bare_mean` / `delta_mean`、各 chunk 系数，以及逐任务的原始 reward、bare reward 和 delta，便于独立复算。Gate manifest 与 selected bank 保存 baseline provenance。

当前仍采用有符号系数降序的 strictly-positive top-K；不足 K 不补齐。Online REWRITE 先做互斥 replacement-family winner selection（同分保留 old），再做全局 positive top-K。正负用于暂定选择规则，并不等于已证明“有益/有害”；大小的排序价值也需要独立 held-out 验证，后续实验见 [ranking validation plan](./trace2skill_ranking_validation_plan.md)。

逐任务相减旨在消除可预测的任务难度差异，但 \(B_i\) 是一次有噪声的观测，不是真实期望值；是否降方差取决于 \(R_i,B_i\) 的协方差及基线噪声，不能保证比原估计更显著。该无截距 main-effect 模型是相对于 bare 的加性近似；存在 chunk 交互、异质性或模型失配时，系数不是自动成立的独立因果效应，也不保证等于随机 mask 分布下的平均边际效应。零系数同样不能证明已内化。

如另做 HC3 或 bootstrap 推断，应对 paired deltas 拟合；task-level bootstrap 联合重采样完整的 `(C_i, R_i, B_i)`，不能把同一 task 的两条 rollout 当独立样本。反复复用同一 bare run 的多轮 masks 也共享基线噪声，联合分析时应按 task 聚类，而不是把它们看作独立证据。
