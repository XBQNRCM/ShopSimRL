# 训练监控与 W&B 记录

本文说明 ShopSimRL 训练外循环的监控口径和操作方式。监控覆盖三个独立阶段：

1. slime GRPO 训练，按 `rollout/step` 连续记录；
2. Failure Analyst，训练期间 sidecar 写 `failure_cards.jsonl`，W&B 仍在 round 结束 compile 后记录一次；
3. randomized validation gate，完成或中止一轮验证后记录一次。

训练轨迹、验证轨迹、回归输入和 gate 结果的本地文件仍是审计依据。W&B 用于查看曲线、横向比较 round 和集中保存阶段产物，不替代 `runs/` 下的原始记录。

## 1. Run 组织

三个阶段使用同一个 W&B project 和 group，各自建立 run：

| 阶段 | run 名称 | `job_type` | 记录频率 |
|---|---|---|---|
| slime 训练 | `WANDB_GROUP` | slime 默认 | 每个 rollout batch |
| Failure Analyst | `<online_analysis.name>-analysis` | `analysis` | 阶段结束一次 |
| online gate | `<online_gate.name>-gate` | `gate` | 阶段结束一次 |

例如第一轮统一设置：

```bash
export WANDB_PROJECT=shopsimrl
export WANDB_GROUP=qwen35-4b-round-000
```

训练脚本传入 `--disable-wandb-random-suffix`，确保 slime 不修改 group 名；Analyst 和 gate 命令使用相同的 `WANDB_GROUP` 后，三个 run 会出现在同一组中。进入下一轮时应同时更新 group，例如 `qwen35-4b-round-001`。

## 2. 认证和启用

在实际训练机器、同一用户和实际 Python/容器环境中安装并登录：

```bash
python -m pip install wandb
wandb login --verify
wandb status
```

不要把 API key 写进仓库，也不要给 slime 传 `--wandb-key`。slime 会把命令行参数复制到 W&B run config；使用 `wandb login` 写入的用户凭据，或由集群 secret 管理器注入的 `WANDB_API_KEY`。当前脚本是单机 Ray；多机训练时，每个节点都必须能读取凭据。

训练侧默认关闭，通过环境变量开启：

```bash
export USE_WANDB=1
export WANDB_PROJECT=shopsimrl
export WANDB_GROUP=qwen35-4b-round-000

# 可选
export WANDB_ENTITY=my-team
export WANDB_MODE=online
export WANDB_DIR=/path/to/persistent/wandb

bash scripts/run_shopsimrl_slime.sh
```

`USE_WANDB` 只接受 `0` 或 `1`。默认值如下：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `USE_WANDB` | `0` | 是否启用 slime 的 W&B logger |
| `WANDB_PROJECT` | `shopsimrl` | W&B project |
| `WANDB_GROUP` | `shopsimrl-training` | 关联同一 round 的多个 run |
| `WANDB_MODE` | `online` | `online`、`offline` 或 `disabled` |
| `WANDB_DIR` | `<project>/runs/wandb` | 训练侧本地 W&B 文件目录 |
| `WANDB_ENTITY` | 空 | W&B team/entity |

即使 `USE_WANDB=0`，自定义 hook 仍会把 ShopSim 指标合并到 slime 的本地日志字典；不会初始化 W&B 或上传数据。

## 3. 训练指标口径

入口为 `shopsimrl.slime_metrics.enrich_rollout_metrics`。该函数不直接调用 `wandb.log`，只扩充 slime 主日志进程的 `rollout_extra_metrics`，随后由 slime 与原生 reward、性能和训练指标一起写一次 W&B。这避免 Ray rollout workers 重复上报。

动态工具 schema 会把一条多轮 trajectory 分成多个 token-contiguous `Sample`。所有段共享 `rollout_id`；监控与 GRPO reward normalization 一样，按 `(group_index, rollout_id)` 去重。因此长轨迹不会因分段较多而在均值中获得更高权重。

### 3.1 结果和覆盖

| 指标 | 定义 |
|---|---|
| `rollout/shopsim/trajectory_count` | 本步**全部已完成生成**（含被动态过滤丢掉的）去重 trajectory 数 |
| `rollout/shopsim/group_count` | 同上，全部已完成 prompt group 数 |
| `rollout/shopsim/train_trajectory_count` | 实际进入 trainer 的去重 trajectory 数 |
| `rollout/shopsim/train_group_count` | 实际进入 trainer 的 prompt group 数 |
| `rollout/shopsim/scored_rate` | 全部已完成 trajectory 中可评分的比例 |
| `rollout/shopsim/reward_mean` | 全部已完成、可评分 trajectory 的主 reward 均值 |
| `rollout/shopsim/strict_reward_mean` | 全部已完成、可评分 trajectory 的 `r_strict` 均值 |
| `rollout/shopsim/success_rate` | 全部已完成、可评分 trajectory 的 `r_success` 均值 |

这些曲线描述本步 SGLang 实际跑完的策略轨迹，包括 over-sampling 后因 zero-std 丢掉、以及凑满 16 组之后多出来的有差异组。中止未完成的 abort 不计入。`train_*` 才是进入 GRPO 的子集。训练 rollout 的 token、logprob、loss mask 和 advantage 仍由 slime 训练数据持有；W&B 标量不承担逐 token 审计。

入口仍是 `enrich_rollout_metrics`；全量 group 由 `--rollout-all-samples-process-path`（`record_generated_rollout_groups`）在过滤前交给它。未接该 hook 时回退为只统计进入 trainer 的样本。

### 3.2 q、rho 和实际 mask

| 指标 | 定义 |
|---|---|
| `rollout/shopsim/q` | 冻结 curriculum 的 skill-free probability |
| `rollout/shopsim/rho_min` | active chunks 中最小 rho |
| `rollout/shopsim/rho_mean` | active chunks 的 rho 算术均值 |
| `rollout/shopsim/rho_max` | active chunks 中最大 rho |
| `rollout/shopsim/skill_free_group_rate` | 全部已完成生成 groups 中实际命中 skill-free 的比例 |
| `rollout/shopsim/assisted_empty_group_rate` | 全部已完成生成 groups 中进入 assisted、但未命中任何 chunk 的比例 |
| `rollout/shopsim/skills_per_assisted_group_mean` | 非 skill-free groups 中注入 chunk 数的均值；包含 assisted-empty |
| `rollout/shopsim/chunk/<chunk-id>/target_rho` | 该 chunk 在 assisted 分支的目标 rho |
| `rollout/shopsim/chunk/<chunk-id>/inclusion_rate` | 该 chunk 在当前 assisted groups 中的实际入选率 |

`inclusion_rate` 的分母只包含全部已完成生成中的 assisted 与 assisted-empty groups，不包含 q 命中的 skill-free groups，因此可以直接和 `target_rho` 比较。单个 batch 的 group 数较少时会有明显 Bernoulli 波动，应观察多个 step 的趋势；它不参与 gate 判定。

### 3.3 all-wrong、retry 和技术失败

| 指标 | 定义 |
|---|---|
| `rollout/shopsim/all_wrong_group_rate` | 在 siblings 全部可评分且 success 数值完整的 groups 中，所有 `r_success=0` 的比例 |
| `rollout/shopsim/full_skill_retry_rate` | 实际触发 full-skill diagnostic retry 的 group 比例 |
| `rollout/shopsim/full_skill_retry_success_rate` | 可评分 full-skill retries 中 `r_success=1` 的比例 |
| `rollout/shopsim/internalization_deficit_rate` | triage 为 `model_internalization_deficit` 的 group 比例 |
| `rollout/shopsim/pending_analysis_rate` | triage 为 `full_skill_failure_pending_analysis` 的 group 比例 |
| `rollout/shopsim/runtime_error_group_rate` | 有 triage 记录的已完成生成 groups 中，标记环境/runtime 错误的比例 |
| `rollout/shopsim/technical_group_drop_count` | 动态采样期间因训练 trajectory 不可评分而丢弃的 group 数 |
| `rollout/shopsim/technical_group_drop_rate` | `drop_count / (accepted_groups + drop_count)` |
| `rollout/shopsim/zero_std_group_drop_count` | 因 sibling reward 全部相同、advantage 恒为 0 而丢弃的 group 数 |
| `rollout/shopsim/zero_std_group_drop_rate` | `drop_count / (accepted_groups + drop_count)` |

训练 trajectory 本身出现技术错误时，整个 group 会从 trainer 过滤并补采。全量生成监控仍包含其已生成 siblings：不可评分 trajectory 影响 `scored_rate`，但不进入 reward/success 均值；同组可评分 siblings 仍进入全量 outcome 均值。这类故障同时应看 `technical_group_drop_*`。`runtime_error_group_rate` 同时覆盖有 triage 的训练或 diagnostic retry 技术错误，不能只解释为 retry 错误。

sibling reward 全部相同的 group 对 GRPO 没有梯度贡献。reward 的比较按 `rollout_id` 去重，fan-out 分段不算多个样本。`over_sampling_batch_size=32` 时第一波有富余，`remaining>16` 的 zero-std 会真丢；只在手里已经不够再采时才放行。所以 `zero_std_group_drop_rate` 是丢弃率而非 zero-std 发生率。波次、`remaining` 和凑批规则见 [dynamic_sampling_filter.md](dynamic_sampling_filter.md)。两个 drop 指标共用 **进入 trainer 的** accepted 分母，各自只计入自己的丢弃数。all-wrong group 在过滤前就已写出 trace 和 triage，Failure Analyst 不受影响。`reward_mean` / `success_rate` 在全部已完成生成上计算，不被筛选偏向 0.5。

## 4. Failure Analyst 上报

Failure Analyst 完成后显式开启 W&B（配置由 `configs/archive/trace2skill_online_analysis.example.yaml` 复制后填写）：

```bash
python scripts/run_shopsimrl.py training analyze \
  configs/trace2skill_online_analysis.yaml \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP"
```

可选参数为 `--wandb-entity`、`--wandb-mode` 和 `--wandb-dir`。相应环境变量也可作为默认值。

上报标量：

- `analysis/triage_groups`
- `analysis/analyzed_cards`
- `analysis/eligible_cards`
- `analysis/candidates`
- `analysis/proposal_history_records`
- `analysis/evidence_only_cards`

同时创建 `shopsimrl-analysis` Artifact，包含存在的 `analysis_summary.json`、`candidate_pool.json`、`failure_cards.jsonl` 和 `proposal_ledger.jsonl`。

`failure_cards.jsonl` 包含 privileged audit。使用 W&B 公有云时，这个文件会随 `--wandb` 上传；需要把 gold-aware 记录限制在训练集群内时，应使用 offline/私有部署，或不要给 Analyst 命令加 `--wandb`，本地 JSON 不受影响。

## 5. Online gate 上报

配置由 `configs/archive/trace2skill_online_gate.example.yaml` 复制后填写：

```bash
python scripts/run_shopsimrl.py training online-gate \
  configs/trace2skill_online_gate.yaml \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-group "$WANDB_GROUP"
```

每次调用记录：

| 指标前缀 | 内容 |
|---|---|
| `gate/complete` | gate 是否完整完成；完整为 1，coverage/运行失败导致的 incomplete 为 0 |
| `gate/validation/*` | requested、completed、failed、scored、coverage 等 masked validation 覆盖统计 |
| `gate/masked/*` | masked reward mean/SEM 和 success rate |
| `gate/masked/reward_component/*` | masked 的各个 `r_*` 均值 |
| `gate/bare/*` | 同 checkpoint、同 task 的 bare mean outcomes |
| `gate/reward/bare_mean` | paired 回归输入中的 bare reward 均值 |
| `gate/reward/masked_mean` | paired 回归输入中的 masked reward 均值 |
| `gate/reward/delta_mean` | `masked - bare` 的样本均值 |
| `gate/intervention_count` | OLS 中的 intervention/version 行数 |
| `gate/positive_chunk_count` | 所有 intervention/version 行中系数严格大于 0 的数量；含 rewrite loser，不等于 survivor 数 |
| `gate/selected_chunk_count` | rewrite family 选择和 positive top-K 后的最终 active chunk 数 |
| `gate/coefficient_mean`、`gate/coefficient_max` | 所有 intervention/version 主 reward 系数的均值和最大值 |

`gate/chunk_contributions` 是 W&B Table，列为：

```text
intervention_id, logical_chunk_id, operation, coefficient, rank, status
```

gate Artifact 包含存在的 `online_gate_manifest.json`、`contributions.json`、`selected_skillbank.json`、`selected_skill.md`、`proposal_ledger.jsonl` 和 `mask_assignments.json`。如果 gate 不完整，只上报已有覆盖率、`gate/complete=0` 和已生成文件，不会伪造回归或 selection 指标。

## 6. Offline 模式

训练机器不能访问 W&B 时：

```bash
export USE_WANDB=1
export WANDB_MODE=offline
export WANDB_DIR=/persistent/path/wandb
bash scripts/run_shopsimrl_slime.sh
```

Analyst/gate 使用：

```bash
python scripts/run_shopsimrl.py training analyze CONFIG \
  --wandb --wandb-mode offline --wandb-dir /persistent/path/wandb
```

任务结束后在可联网机器执行：

```bash
wandb sync /persistent/path/wandb/wandb/offline-run-*
```

不要把离线目录放在容器临时层；确认它与 checkpoint、`runs/` 一起持久化。

## 7. 建议面板与判读

建议为每个 group 建立以下面板：

1. **学习结果**：`reward_mean`、`strict_reward_mean`、`success_rate`；
2. **课程采样**：实际 `skill_free_group_rate` 对照 q，各 chunk `inclusion_rate` 对照 `target_rho`；
3. **failure frontier**：`all_wrong_group_rate`、`full_skill_retry_success_rate`、`pending_analysis_rate`；
4. **运行健康**：`scored_rate`、`technical_group_drop_count/rate`、slime 原生 rollout time 和吞吐；
5. **round gate**：bare/masked/delta reward、selected count 和 chunk contribution Table。

优先检查以下信号：

- `technical_group_drop_count > 0`：先查看相应训练 trace 的环境、parser 或模型服务错误；这些不是 policy failure；
- `scored_rate < 1`：全量生成中存在不可评分轨迹，需检查技术错误及丢弃补采情况；不等于 trainer 已接受不可评分样本；
- 实际 skill-free rate 长期偏离 q，或多个 chunk inclusion rate 长期偏离 rho：检查 curriculum identity、group key 和 mask 一致性；
- `all_wrong_group_rate` 上升但 full-skill retry 大量成功：模型内化不足的证据增强；
- pending analysis 上升且 eligible cards/candidates 始终为 0：检查 Analyst counterfactual trial、firewall 和 compiler 输出；
- gate coverage 小于 1：只能续跑补齐验证，不应读取 selected count 作为有效结果。

训练中的 reward 与 mask 曲线是监控信号。chunk 是否进入下一轮仍只由固定 checkpoint 的 paired-delta validation gate 决定。

## 8. 启动前检查

```bash
wandb status
python -c "import wandb; print(wandb.__version__)"
bash -n scripts/run_shopsimrl_slime.sh
```

训练开始后确认日志中同时出现 slime 原生 `rollout/*` 和 `rollout/shopsim/*`。第一批完成后，W&B 页面至少应看到：

- `rollout/shopsim/q`；
- `rollout/shopsim/reward_mean`；
- `rollout/shopsim/skill_free_group_rate`；
- 每个 active chunk 的 `target_rho` 与 `inclusion_rate`。

W&B 上传失败不会改变已经原子写入的 ShopSimRL trajectory/triage JSON；online 模式下 logger 异常是否中止训练由 W&B/slime 客户端行为决定。正式长跑前应在少量卡 smoke 中验证一次 online 或 offline run、页面字段、Artifact 内容和磁盘持久化路径。
