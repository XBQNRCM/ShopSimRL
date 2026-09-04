# Trace2Skill contribution 排序价值：后续实验计划

记录日期：2026-09-03。状态：**待执行**。本文件只记录后续实验，不启动任务，不改变现有 chunk selection 或覆盖已有报告。

## 1. 研究问题与当前进度

目标是验证 contribution 能否作为 **chunk 排序/预算选择的启发式**，暂不要求其符号准确表示独立作用方向，也不要求系数数值准确预测完整 skill 的 reward 增量。

这里的“按系数大小排序”指按有符号的 `beta` 从大到小排序，不是按 `abs(beta)` 排序。

当前回归设计已经从带自由截距的 `R_i = intercept + C_i^T beta + epsilon_i` 改为逐任务 paired-delta 无截距回归：

```text
B_i = 同一个 validation task 的 bare reward
R_i = 该 task 的 masked-skill reward
C_i = chunk mask
Delta_i = R_i - B_i
Delta_i = C_i^T beta + epsilon_i  # 不设 intercept
```

这是相对 bare 的加性贡献近似。存在 chunk interactions、共同的 skill prompt 开销或服务分布差异时，系数不自动等同于稳定的平均边际因果效应；系数和也不一定等于实际 bundle 效果。

该方法现已统一为 cold-start 与训练 online gate 的默认实现；同 checkpoint 的 bare validation traces 通过 `bare_run_dir` 复用，配对契约见 [paired_validation.md](./paired_validation.md)。下列路径保留为历史探索实验记录，后续受控实验使用严格匹配的数据；这里列出的排序验证实验仍待补做。

已有数据：

- 初始 16-chunk SkillBank：`runs/qwen35-4b-train-0830/trace2skill-cold-start/initial_skillbank.json`。
- Gate A masks、traces 和原回归：`runs/qwen35-4b-train-0830/trace2skill-cold-start/gate-a/`。
- 逐任务 bare validation：`runs/qwen35-4b-val/`；**不要误用** test bare `runs/qwen35-4b-test-0830/`。
- 新分组为 10 个正系数 chunks、6 个负系数 chunks；已有 bundle 对比：`runs/qwen35-4b-test-paired-delta-positive-vs-negative/comparison.md`。
- 现有 masked validation 来自 SiliconFlow，bare validation 和新 bundle test 来自本地服务；现有重算采用了“服务分布等价”的显式假设。后续实验优先将 bare、mask、bundle 全部统一到同一本地部署。

现有主奖励证据尚不充分：原报告 positive 比 negative 高约 2.92pp，但 95% CI 跨 0；这只是两个完整 bundles 的比较，不能证明逐 chunk 排序有效。

### 现有 negative 重复轨迹问题

`runs/qwen35-4b-test-paired-delta-negative/retry_log.json` 记录了两个 runner 同时写同一目录，387 个任务各完成两次。它不是普通 failed 补测。

按完成顺序取 first/latest 会选择不同的随机 rollout，并可能引入与轨迹时长相关的选择偏差：

| Negative 记录口径 | Negative r_strict | Positive − Negative |
|---|---:|---:|
| 第一条 completed | 0.572720 | +0.005294 |
| 最后一条 completed（原报告） | 0.548838 | +0.029176 |
| 同任务 completed 取平均（敏感性分析） | 0.560779 | +0.017235 |

任务内平均口径下，主奖励差值的 task-level bootstrap 95% CI 约为 `[-0.02387, +0.05823]`，步数和 token 优势也不再显著。此平均只是敏感性分析，不将意外重复视为规范设计的重复采样。双进程竞争资源还会干扰延迟比较。

## 2. 共同前提与数据规范

- [ ] 冻结同一本地模型部署、checkpoint/权重版本、tokenizer/chat template、量化设置、context cap、解码参数、system prompt、runtime 和环境版本，并记录 provenance。
- [ ] 两组/多组使用相同执行并发；避免同一推理服务的额外负载干扰成本与延迟比较。
- [ ] 同一 run 目录只允许一个 runner。建议启动时使用进程锁，记录 `run_id`、`attempt_id` 和进程信息。
- [ ] failed 重试与独立重复采样分开：重试沿用 episode，已 complete 的任务不重跑；重复实验使用新的 `sample_id` 或独立 run ID。
- [ ] 不按完成先后或 reward 挑选重复轨迹；在执行前冻结纳入和聚合规则。
- [ ] 使用原始 chunk 内容与 canonical draft order，不改写/删减 `Applies when`、`Procedure`、`Verify`、`Avoid`。
- [ ] `skills.max_skills: null`，避免注入内容被截断；逐条检查实际 selected chunk IDs。
- [ ] 冻结主指标 `reward/r_strict`，`r_success` 为重要辅助指标；报告全部条件的 coverage 和失败原因。

### 数据独立性

当前 beta 使用了全部 400 个 val tasks，不能事后把其中一半改称其独立 holdout。可选方案：

1. 冻结现有 beta，在尚未用于拟合/选择的新 validation tasks 上评估。
2. 如果没有新任务，先将 validation tasks 固定划分为 fit/holdout 两部分，只用 fit 部分**重新估计 beta 和选择 chunks**，在 holdout 部分评估。拟合部分仍可复用已有 paired observations。

只在同一批任务上换 seed 或 mask，验证的是重复 rollout/新 mask 下的稳定性，不等同于对新任务的泛化。两种目标需分别标注。

后续方法探索优先在独立 validation 上进行；不要连续根据已看过的 test 结果调规则，再宣称该 test 是 untouched held-out evaluation。

## 3. 实验 A：同预算 Top-K / Random-K / Bottom-K

### 假设

若 beta 具有实际排序价值，相同 skill 预算下，高分组应较低分组有更好的 held-out reward；Random-K 提供随机选择基线。

### 设计

- [ ] 预先固定一个主 `K`（可从 `K=4` 开始；最终值执行前确定），不要看结果后挑 K。
- [ ] `Top-K`：有符号 beta 最高的 K 个 chunks。
- [ ] `Bottom-K`：有符号 beta 最低的 K 个 chunks。
- [ ] `Random-K`：从 16 个 chunks 均匀无放回抽 K 个；每个随机 subset 在所有评测任务上保持固定。
- [ ] Random-K 使用若干预先冻结的 subset seeds（例如 5 组，按预算确定），避免用一个偶然随机 bundle 代表随机选择方法。
- [ ] 若预算允许，加一组同服务、同任务的 Bare，用于判断每组相对 no-skill 的收益。
- [ ] 所有条件使用相同任务和明确的 rollout sample IDs；必要时增加重复采样。

相同 K 不保证相同 prompt token 数。需记录实际注入 token 长度；可以另加预先定义的长度匹配敏感性实验，但不要为了匹配长度重写 chunk，或把受长度约束的选择混称为纯 Top-K。主 Top-K 规则与长度匹配规则应分开报告。

### 分析与判读

- [ ] 报告 Top-K − Bottom-K、Top-K − Random-K、Random-K − Bottom-K 的 task-paired reward 差值和 95% CI。
- [ ] Random-K 同时报告不同 subset seeds 的结果分布，不能只挑一组；聚合时任务等权。
- [ ] Bootstrap 保留同一任务内的所有 arms/rollouts；多个 rollout 不视为独立 task。若要泛化到随机 subset 的分布，还需反映 subset 间不确定性。
- [ ] 主 reward 对比之外，报告 success flips、reward components、protocol/invalid action、步骤和 tokens；多重检验规则提前指定。

若能在独立数据及重复实验上稳定观察到 Top-K 优于随机和低分选择，即使系数未校准，也支持其作为选择器的价值。点估计的单次顺序不足以确证这一结论。

## 4. 实验 B：冻结 beta，在新 masks 上验证预测排序

### 假设

冻结 beta 后，预测分数较高的 mask 应倾向于产生较高的实际 reward 增量：

```text
predicted_score_i = C_i^T beta_frozen
observed_delta_i = R_i(mask) - B_i
```

### 设计

- [ ] 按第 2 节选择独立 holdout；测试期间不更新 beta。
- [ ] 优先使用恰好 K 个 chunks 的随机 masks，排除“chunk 数量更多导致分数更高”的简单混杂。
- [ ] 如果考察多个 K，在每个 K 内分别检验，或预先设计分层分析；不要仅靠跨 K 的总体相关性宣称 chunk 排序有效。
- [ ] 冻结 mask seeds，并保存任务、sample ID、mask、预测分数、实际注入长度和 reward。
- [ ] 为同一任务收集 bare 和 masked outcomes；同一冻结 checkpoint/config 内可以复用 bare 记录，跨 checkpoint 不复用。
- [ ] 如果每个 task 有多个 masks，保留其共同 bare baseline 带来的相关性，使用 task-cluster bootstrap。

### 分析与判读

- [ ] 主排序指标：`predicted_score` 与 `observed_delta` 的 Spearman 相关及 task-level/cluster bootstrap CI。
- [ ] 辅助展示预测分数分位数组的实测平均增量，不用分位组结果重新修改 beta。
- [ ] 若同一任务有多个 masks，可补充任务内排序一致性，减少剩余任务难度差异的影响。
- [ ] 另报预测值与实测值的校准斜率/误差，用于区分“排序有效”和“绝对数值预测准确”。这些诊断不在 holdout 上用于重新拟合最终选择规则。

显著正向、可重复的排序相关性支持 beta 的排序价值；校准差但排序稳定时，应使用 rank/预算选择，不把系数和解释为预期实际 reward 增量。

## 5. 建议执行顺序与交付

1. 先修复 run 隔离和重复轨迹规范，保留原报告及敏感性分析，不覆盖历史数据。
2. 明确独立 validation 来源、冻结拟合数据和本地服务配置。
3. 优先执行一个固定 K 的实验 A，验证最直接的选择器价值。
4. 有预算时执行实验 B，直接检验分数的排序能力与校准程度。
5. 最后才根据预先定义的规则冻结新方法，并在合适的独立数据上报告最终结果。

后续交付清单：

- [ ] 独立 YAML、冻结 beta/selection、mask/subset seeds 和完整 manifest。
- [ ] 每个 arm/seed 的完整 traces、summary 与重试记录。
- [ ] `scripts/` 下可复现的独立分析脚本；当前计划不要求修改核心 evaluator。
- [ ] task-aligned comparison JSON/Markdown、相关性与置信区间。
- [ ] 明确区分排序、数值校准、相对 bare 的作用方向、资源成本四类结论。
