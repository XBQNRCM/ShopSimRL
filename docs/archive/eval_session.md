# 评测实验临时备忘录（2026-09-07）

训练已停在 round-003 / iter 80。本文只排本 session 要做的事，不改 selection 规则，不重训。做完一项勾一项。

## 0. 冻结口径

所有新评测共用训练循环 val/gate 的协议，本机 sglang，不用 SiliconFlow。

| 项 | 值 |
|---|---|
| 环境 | `http://127.0.0.1:5700`，48 槽 |
| 模型服务 | `http://127.0.0.1:30000/v1`，`--tp 1 --dp-size 4`，`tool-call-parser qwen3_coder`，`reasoning-parser qwen3` |
| 采样 | `temperature=0.6`，`top_p=1.0`，无 `top_k`，`max_tokens=4096`，`enable_thinking: true`，`tool_choice: auto`，`send_seed: false`，`trust_env: false` |
| runtime | `max_steps=30`，`concurrency=32`，`seed=20260830`，`repeats=1`，`max_skills: null` |
| 任务 | `persona_splits.v1.json` 的完整 `test` 400；ranking 拟合 beta 才用 `val` 400 |
| 主指标 | `reward`/`r_strict`；辅助 `r_success` 与各 `r_*`；必须 `coverage=1` |

权重与 skill（只读，不改内容）：

| 角色 | 路径 | `checkpoint_id` |
|---|---|---|
| 基座 | `/root/autodl-tmp/models/Qwen3.5-4B` | `qwen35-4b-base-hf` |
| 终局 | `runs/slime-qwen35-4b-hf`（iter 80） | `qwen35-4b-iter80` |
| 初始 skill \(S_0\) | `runs/qwen35-4b-train-0830/trace2skill-cold-start/paired-delta-recovered/selected_skillbank.json`（10 chunk） | — |
| 终局 skill \(S_T\) | `runs/round-003-online-gate/selected_skillbank.json`（8 chunk） | — |
| ranking 因子 | `runs/qwen35-4b-train-0830/trace2skill-cold-start/initial_skillbank.json`（16 draft chunk） | — |

同一时刻只挂一份权重。换权重必须换 `experiment.name` 和 `checkpoint_id`，并在 bare/equipped/补测期间保持服务冻结。
0830 SiliconFlow 的 val/test（2048 / `top_p=0.95` / `top_k=20`）不与本表数字混画。

对比一律用 `scripts/compare_shopsimrl_runs.py`（task-aligned bootstrap）。历史 positive vs negative 因双进程重复写目录，不作正式证据。

## 1. 目录：先清 round-4，再归档被覆盖的旧评测

未完成的第 5 轮直接删，不进 archive。不要误删 HF 分片 `model-00004-of-00019.safetensors`。

删除：

```text
runs/slime-training/round-004/
runs/round-004-online-analysis/
runs/round-004/
runs/loop/round-004/
```

旧评测 config 将被新本地协议覆盖；对应 runs 先搬到 `runs/archive/`，否则新实验若复用目录会撞 fingerprint。

```text
runs/archive/qwen35-4b-test-0830/
runs/archive/qwen35-4b-val/
runs/archive/qwen35-4b-test-paired-delta-positive/
runs/archive/qwen35-4b-test-paired-delta-negative/
runs/archive/qwen35-4b-test-paired-delta-positive-vs-negative/
```

不动：`round-000`～`003` 全套、`slime-training/round-000`～`003`、`slime-qwen35-4b`、`slime-qwen35-4b-hf`、`qwen35-4b-train-0830/`（含 \(S_0\) 与 16-chunk draft）、`snapshots/`。

随后覆盖这些 yaml（内容对齐 §0，实验名如下）。`qwen35_4b_train.yaml` 与 `trace2skill_*.yaml` 先不动。

| 文件 | 新用途 | `experiment.name` |
|---|---|---|
| `configs/qwen35_4b_test.yaml` | 基座 skill-free test | `qwen35-4b-test-base-free` |
| `configs/qwen35_4b_test_trace2skill_equipped.yaml` | 基座 + \(S_0\) | `qwen35-4b-test-base-s0` |
| `configs/qwen35_4b_test_final_free.yaml`（新） | 终局 skill-free test | `qwen35-4b-test-iter80-free` |
| `configs/qwen35_4b_test_final_pair.yaml`（新） | 终局 + \(S_T\) | `qwen35-4b-test-iter80-st` |
| `configs/qwen35_4b_val.yaml` | 基座 local bare val（给 ranking 拟合） | `qwen35-4b-val-base-free` |
| `configs/qwen35_4b_val_slime.yaml` | 本地评测模板（4096 / thinking / `top_p=1.0`） | 模板，不直接跑 |
| `configs/qwen35_4b_test_trace2skill_negative.yaml` | ranking Bottom-K 占位，等 §5 冻结 bank 再填 | 暂不跑 |

`skills.path` 只允许 \(S_0\) / \(S_T\) / ranking bank；`max_skills` 必须 `null`。

## 2. 实验顺序（一个一个做）

### A. 基座 test：skill-free 与 \(S_0\)

服务基座 HF。先 `plan` 再 `run`。

1. `qwen35-4b-test-base-free`：无 skill。得到 \(R_{\mathrm{free}}^{\mathrm{base}}\)。
2. `qwen35-4b-test-base-s0`：注入冻结 \(S_0\)。得到 \(R_{\mathrm{pair}}^{\mathrm{base}}(S_0)\)。

对比：\(R_{\mathrm{pair}}^{\mathrm{base}}(S_0)-R_{\mathrm{free}}^{\mathrm{base}}\) = **Trace2Skill（冷启动 skill）在基座上的增量**。

### B. 终局 test：skill-free 与 \(S_T\)

停 sglang，改挂 `runs/slime-qwen35-4b-hf`。

3. `qwen35-4b-test-iter80-free`：无 skill。得到 \(R_{\mathrm{free}}^{T}\)。
4. `qwen35-4b-test-iter80-st`：注入冻结 \(S_T\)。得到 \(R_{\mathrm{pair}}^{T}\)。

对比：

- \(R_{\mathrm{free}}^{T}-R_{\mathrm{free}}^{\mathrm{base}}\)：参数内化（同协议基座对照）
- \(R_{\mathrm{pair}}^{T}-R_{\mathrm{free}}^{T}\)：终局 internalization gap
- \(R_{\mathrm{pair}}^{T}-R_{\mathrm{free}}^{\mathrm{base}}\)：共同演化相对裸基座的上界
- \(R_{\mathrm{pair}}^{T}\) vs \(R_{\mathrm{pair}}^{\mathrm{base}}(S_0)\)：模型已换、skill 已换，只作描述、不作因果

### C. contribution 排序（ranking）

不改历史 \(S_0\)。旧 Gate A 是 SiliconFlow / 可能带截距，**不复用那份 beta**。在本地基座上重新拟合，只在 test 上评估排序。

预先冻结、看结果前不得改：

- \(K=4\)（有符号 beta 从大到小，不是 `|beta|`）
- Random-4 原计划 5 个 subset seed：`20260907`…`20260911`；**本 session 只跑 `20260907`**，其余 seed 的 yaml 保留但不跑
- Top-4 / Bottom-4 来自**本机** paired-delta Gate A，不是 recovered 的 10-chunk \(S_0\)

步骤：

1. [x] 再挂基座。跑 `qwen35-4b-val-base-free`（400 bare val）。
2. [x] 对 16 draft chunk 跑本地 Gate A（masked val，复用上一步 bare）。产物写新目录 `runs/qwen35-4b-val-base-gate-a/`，不要写进历史 `gate-a/`。
3. [x] 按该 beta 冻结 bank：Top-4、Bottom-4、Random-4（`20260907`）。
4. [x] 在 test 上跑 Top / Bottom / Random-`20260907`。Bare 复用 A.1，不重跑。Random 曾欠费中断，已 resume 至 coverage=1。
5. [x] 报告 Top−Bottom、Top−Random、Random−Bottom 的 task-paired 差与 95% CI（单 seed）。对比见 `runs/qwen35-4b-val-base-gate-a/ranking_test_comparison.json`。

实验 B（新 mask 上 `C^T beta` 与实测 delta 的 Spearman）有预算再做，本备忘录不排进第一批。

## 3. 建议动手顺序

1. 删 round-4 残尾，archive 旧评测 runs，覆盖/新增 yaml。
2. 起 ShopSimulator + 基座 sglang，做 A.1、A.2。
3. 换终局权重，做 B.3、B.4，写四组对比。
4. 再挂基座，做 ranking（C）。

当前 ShopSimulator 与基座 sglang 仍在跑（补 Random 后未关）。

## 4. 本 session 数字（本地协议，test 400，coverage=1）

主实验：

| run | r_strict | success |
|---|---:|---:|
| `qwen35-4b-test-base-free` | 0.5422 | 0.5175 |
| `qwen35-4b-test-base-s0` | 0.5663 | 0.545 |
| `qwen35-4b-test-iter80-free` | 0.8501 | 0.8425 |
| `qwen35-4b-test-iter80-st` | 0.8671 | 0.8625 |

Ranking（基座；K=4；Random 仅 seed `20260907`）：

| arm | r_strict | success | vs free Δreward [95% CI] |
|---|---:|---:|---|
| Top-4 | 0.5417 | 0.5175 | −0.0004 [−0.043, +0.041] |
| Random-4 | 0.5589 | 0.5425 | +0.017 [−0.024, +0.058] |
| Bottom-4 | 0.5704 | 0.550 | +0.028 [−0.016, +0.073] |

Top−Bottom Δreward = **−0.029**，CI 跨 0 `[−0.070, +0.011]`。点估计与 val beta 排序相反，但 95% CI 不能拒绝零。
