# 动态采样过滤器

入口：`shopsimrl.slime_runtime.fully_scored_group_filter`。slime 在 `slime/rollout/sglang_rollout.py` 的 `generate_rollout_async` 里调用它。过滤发生在**一组 8 条 sibling 全部生成并打分之后**，判断本身几乎不耗时；已经付过的生成时间不会退回。

波次都在**同一个 slime rollout** 内。过滤决定留下哪些 group，不改变随后的训练切分：留下 16 组共 128 条 episode 后，`global_batch_size=64` 会把它们切成两次优化器更新。

## 1. 当前旋钮

`run_full_train.sh` 的默认值：

| 项 | 默认 | 作用 |
|---|---|---|
| `rollout_batch_size` | 16 | 本步最终要留下的 group 数 |
| `n_samples_per_prompt` | 8 | 每个 group 的 sibling 数 |
| `over_sampling_batch_size` | 32 | 每一波提交多少个 group |
| `global_batch_size` | 64 | 每次优化器更新的 unique episode 数 |

第一波提交 32 组，目标只留 16 组，所以有 16 个名额的富余用来丢掉 zero-std。`run_shopsimrl_slime.sh` 单独跑时，`over_sampling_batch_size` 仍默认等于 `rollout_batch_size`。

## 2. 两种丢弃

`fully_scored_group_filter` 按顺序判定：

1. **技术故障 / 不可评分**（`shopsim_unscored_technical_group`）：`keep=False`，没有兜底。整组丢掉并补采，不会以 reward=0 进 GRPO。连续失败达到 `shopsim_max_consecutive_failed_groups`（默认 8）则直接终止。
2. **zero-std**（`shopsim_zero_std_group`）：sibling 的 unique-`rollout_id` reward 全相同（全对、全错、或全是同一个中间值）。GRPO advantage 恒为 0。`keep=False`，但 `keep_when_insufficient=True`。

有差异（`max(reward) - min(reward) > 1e-6`）则留下。比较按 `rollout_id` 去重，fan-out 分段不算多个样本。

all-wrong group 在过滤前已经写出 trace 和 triage，Failure Analyst 不受影响。

## 3. `remaining` 和是否再采一波

`remaining_batch_size`（下称 `remaining`）= **已提交组数 − 已丢掉的组数**。留下的组不减 `remaining`。

- 目标 `target = 16`。
- 外层：`len(data) < 16` 就继续。
- 内层：`remaining < 16` 就再提交一整波 `over_sampling_batch_size`（32）个 group。
- 全部在飞任务处理完时：`remaining == len(data)`。

slime 是否丢掉一个 `keep=False` 的组，看 `should_drop_dynamic_filter_output`：

- 普通留下：不丢。
- `keep_when_insufficient=True` 且 `remaining <= 16`：不丢（放行进训练）。
- 其余：丢掉，`remaining -= 1`。

因此第一波提交 32 组后 `remaining=32`：

- **有差异的组留下**；`data` 到 16 后，其余完成的组即使有差异也不进训练，还在生成的会被 abort。
- **全 0/全 1 且 `remaining > 16` 会真丢**，这是 over-sampling 真正换掉零优势组的窗口。
- 丢掉 16 组之后 `remaining=16`，再遇到的全 0/全 1 走兜底，避免再开一波。
- 技术故障始终必丢。只有 `remaining` 掉到 16 以下才会再提交一波 32 个新 prompt，不是同一组重跑。

## 4. 第二波之后怎么凑 16

第二波只在第一波丢掉太多（主要是技术故障，或 zero-std 把 `remaining` 削到 16 以下）时出现。提交的瞬间 `remaining` 会再次大于 16。

之后谁先完成谁先判：

| 完成时的组 | 行为 |
|---|---|
| 有差异 | 留下；`data` 不满 16 就写入 |
| 全 0/全 1，且 `remaining > 16` | 真丢，`remaining -= 1` |
| 全 0/全 1，且 `remaining <= 16` | 放行，避免再开一波 |
| 技术故障 | 必丢 |

**不是**按「谁更有差异」从在飞组里重排。已经进 `data` 的全 0/全 1 **不会被后完成的组踢掉**。后一波补的是名额。

例：第一波 32 组里留下 16 组（14 有差异 + 2 兜底全 0），训练 batch 就是这 16 组，不会开第二波。被丢掉的 zero-std 计入 `zero_std_group_drop_count`。

## 5. 会不会有第三波

- 前两波加起来，**能打分的**（有差异 + 全 0/全 1）≥ 16：用兜底把 `data` 填到 16，**没有第三波**。其中可以包含 advantage 为 0 的组。
- 能打分的 < 16（技术故障丢得太多）：`remaining` 会掉到 16 以下，**会开第三波、第四波**，直到留下 16 组，或连续失败终止。

有 over-sampling 时，全 0/全 1 可以把 `remaining` 往下削，但削到 16 就会改走兜底，单独不会无限补采。

## 6. 对时间和梯度的影响

- 过滤判断本身不加 rollout 时间。
- 第一波固定提交 32 组，rollout 墙钟大约按 32 组而不是 16 组计；训练仍吃留下的 128 条 episode。
- `global_batch_size=64` 把这 128 条切成两次更新（各 8 个完整 group），不改变过滤留下的集合。
- `zero_std_group_drop_count` 是**实际丢掉**的组数。over-sampling=32 时，第一波里 `remaining>16` 的 zero-std 会进入这个计数。
- 两个 drop 指标共用 accepted 分母，各自只计入自己的丢弃数。
- 筛选后留下的 16 组仍偏向有差异；`rollout/shopsim/success_rate` / `reward_mean` 改在全部已完成生成上计算，不再被这 16 组带着走。`train_group_count` 才是进入 trainer 的组数。

相关 W&B 字段见 [training_monitoring.md](training_monitoring.md) §3.3。
