# 基于 slime 的训练链路

当前实现把 proposal 中的快、慢两层时间尺度分开：slime 负责同一 checkpoint 内的 GRPO rollout/update；Trace2Skill failure analysis 与 randomized validation 在一个有限训练 round 结束后运行。两条链路共享冻结、带哈希的 curriculum / candidate-pool / assignment 状态，不修改 slime 核心代码。

## 1. 数据与初始 curriculum

训练只读取项目冻结的 persona `train` task IDs：

```powershell
python scripts\run_shopsimrl.py training prepare-data `
  ShopSimulator\shop_env\configs\persona_splits.v1.json `
  train data\shopsim_train.jsonl
```

冷启动使用最新 paired-delta validation 入选的 10 条 chunk。当前工作区的历史 `gate-a` 是旧的 masked-only 带截距估计（7 条入选），不能作为本次初始化。最新 selection 的 ID、完整文本和系数已从 positive run 的全部 422 条 context 记录中核对恢复，保存至 `paired-delta-recovered/selected_skillbank.json` 和 `selected_skill.md`；记录的文本同时与冷启动 draft 核对一致。恢复脚本只读取已冻结的 skill context，不读取 test reward/success 来重选：

```powershell
python scripts\recover_paired_skillbank.py `
  --comparison runs\qwen35-4b-test-paired-delta-positive-vs-negative\comparison.json `
  --positive-run runs\qwen35-4b-test-paired-delta-positive `
  --draft-skillbank runs\qwen35-4b-train-0830\trace2skill-cold-start\initial_skillbank.json `
  --output-dir runs\qwen35-4b-train-0830\trace2skill-cold-start\paired-delta-recovered
```

`recovery_manifest.json` 保留原 run 的 SkillBank identity 和恢复产物的新 hash；它不是重新运行的 gate manifest。本地没有配置中 `qwen35-4b-val-thinking-tools-v1` 的 bare validation，已有 `qwen35-4b-val` 的 model identity 也不匹配历史 masked run，因此没有绕过校验重拟合，也没有把恢复目录伪装成 `gate-a-paired` 的完整实验产物。

随后从这份 active SkillBank 冻结第一轮 curriculum（pre-validation draft 不能用于训练）：

```powershell
python scripts\run_shopsimrl.py training build-curriculum `
  runs\qwen35-4b-train-0830\trace2skill-cold-start\paired-delta-recovered\selected_skillbank.json `
  runs\qwen35-4b-train-0830\trace2skill-cold-start\paired-delta-recovered\curriculum.json `
  --round-id round-000 `
  --skill-free-probability 0.20 `
  --rho-min 0.20 `
  --rho-max 0.90
```

`q`、`rho_min`、`rho_max` 是显式实验参数。Proposal 没有规定唯一的 q controller，因此代码不把某个未经验证的自动控制器写死。默认 rho 映射是记录在 state 中的 clipped linear mapping；不传 `--contribution-scale` 时，本轮最大正贡献映射到 `rho_max`。Gate 仍独占 chunk admission/retirement 权限，curriculum builder 不会重新接纳非正贡献内容。

每个 curriculum JSON 同时嵌入 active chunks、contribution、rho、q、来源 SkillBank hash 与映射参数。rollout worker 会校验 `state_id`，文件内容被修改但 hash 未更新时直接失败。

2026-09-03 的链路检查已生成上述首轮输入：3726 个冻结 train task、10 个 chunks，q=0.20、rho 范围 [0.20, 0.90]。这组参数沿用本文示例，用于后续 smoke 准备，正式算法讨论后可以重新冻结。`data/` 和 `runs/` 被 Git 忽略，换训练机器时需要同步这些产物或运行上述命令重建；不能只同步代码。

## 2. slime rollout 与 GRPO 语义

训练入口是：

```bash
export HF_CHECKPOINT=/path/to/Qwen3.5-4B
export TRAIN_LOAD=/path/to/qwen35-4b-actor-torch-dist
export TRAIN_SAVE=/path/to/output
bash scripts/run_shopsimrl_slime.sh
```

当前训练与 rollout 均为 TP=2，所以 `NUM_GPUS` 必须是 2 的正整数。脚本同时把
该值传给 slime 的 actor 和 rollout 节点拓扑；2、4、8 卡会分别启动 1、2、4 个
双卡 rollout engine。少于 8 卡时不能只修改 Ray 的可见 GPU 数而保留 slime
默认的 `num_gpus_per_node=8`。

运行前按 ShopSimulator 文档启动环境服务，并按机器拓扑审核 GPU、batch、context 和 checkpoint 路径；脚本会在启动 Ray 前检查 checkpoint、task data、slime checkout 和 custom config，并执行以下无 GPU、无 API 的输入校验。自定义参数位于 `configs/slime_shopsimrl.yaml`；其中 `shopsim_skillbank_path` 与 `shopsim_curriculum_path` 必须对应同一版本：

```powershell
python scripts\run_shopsimrl.py training check configs\slime_shopsimrl.yaml
```

检查覆盖 curriculum 内容 hash、source bank、chunk 文本/系数、完整且不重复的 frozen train IDs、persona 和并发配置。`ready` 只表示离线输入通过，不表示 GPU、SGLang 或环境服务已验证。脚本使用 slime 原生扩展点：

- `shopsimrl.slime_runtime.generate`：多轮 function-tool agent rollout；
- `shopsimrl.slime_runtime.normalize_grpo_by_prompt_and_rollout`：按 prompt group、unique rollout 做 GRPO centering；
- `Sample.tokens` / `rollout_log_probs` 来自 SGLang 实际采样，不从文本重新分词恢复；
- 环境 observation、tool result 和 prompt token 的 loss mask 为 0，模型生成 token 的 loss mask 为 1；
- 动态工具 schema 造成 token-contiguous trajectory 分段时，siblings 共享 `rollout_id`。reward normalization 先按 unique rollout 计算，再向 siblings 广播，不会让长轨迹在组均值中被重复计数。
- 适配器明确使用 `fork_threshold_tokens=0`：slime 默认会在短 assistant reasoning 回显或 token drift 时丢弃先前 response 的训练信号；本实现保留每一轮实际生成的 token/logprob，必要时独立分段。
- 任一 rollout 因环境、协议运行时或空 trajectory 而不可评分时，动态采样过滤器丢弃整个 prompt group 并补采；技术失败不会作为 reward=0 样本污染 GRPO。连续失败达到配置阈值时直接终止，避免环境服务宕机后无限补采。
- SGLang 的 `abort` 及 rollout 停止状态属于技术中止；停止后不再启动新的 turn/retry 请求。达到生成长度上限仍是可评分的 policy failure。当前页面未提供的工具与 evaluator 一样返回 `unavailable_tool` feedback，不执行环境动作。

脚本给 slime dataset loader 开启 `--apply-chat-template`，兼容 Qwen3.5 checkpoint 附带 processor 时必须使用 message-list 的输入要求；实际购物 prompt 仍由 runtime 根据 `metadata.task_id` 从环境生成。`GLOBAL_BATCH_SIZE` 默认等于 `ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT`（原默认仍为 128），缩小 smoke batch 时随之缩小；显式覆盖时必须整除一轮的 episode 数，避免训练侧丢弃尾部 rollouts。

当前入口固定 `top_p=1.0, top_k=-1`。这是 runtime contract，而不是随意的解码偏好：slime 的普通 rollout 能在 `top_p<1` 时携带截断分布 replay 元数据，但当前多轮 `TrajectoryManager` 只保存 exact token/logprob，不能把该 ragged 元数据跨 turn/fan-out 无损传到训练侧。验证配置使用同一采样分布；若未来扩展 manager 的 replay contract，再联合修改训练和 validation，不能只改脚本参数。

`shopsim_episode_concurrency`（正整数，默认 16）限制每个 rollout worker 中同时执行的完整 episode，普通 rollout 与 full-skill retry 共用名额。超额任务在异步 semaphore 上等待，不进入阻塞 `reset` 的线程池；名额覆盖 reset、交互和 close，在 group 协调前释放。该值不能超过分配给此 worker 的环境槽位数；多个 worker 或评测进程共享服务时，需要分别预算，不能各自按服务总槽位数配置。

### Group-consistent 双层 mask

同一个 prompt 的 `n_samples_per_prompt` 个样本共享 slime `group_index`。runtime 先用 `(curriculum state_id, group_index)` 对 q 做一次确定性采样：

- 命中 skill-free：整个 group 不注入任何 chunk；
- 否则进入 assisted，再按每个 chunk 的 rho 采样一次联合 mask。

group 内每个 rollout 都校验 `state_id` 和完整 `skill_ids` 相同。worker 调度、并发完成顺序和断点恢复不会改变 treatment。`assisted_empty` 被显式记录，表示进入 assisted 分支后所有 Bernoulli chunk 均未命中；它不会被伪装成另一次 q 命中。

## 3. all-wrong full-skill retry 与 error analysis

每个训练 rollout 都写入：

```text
runs/slime-training/<round-id>/group-XXXXXXXXX/
├── rollout-<sample-index>.json
├── triage.json
└── full-skill-retry.json      # 只在 all-wrong 时存在
```

只有当 group 中所有 rollout 都是正常、可评分终局且 `r_success=0` 时，runtime 才使用完整 current skill 额外生成一条诊断 trajectory。该 retry：

- 不返回给 slime trainer；
- 不参与原 group 的 GRPO advantage；
- 成功时标为 `model_internalization_deficit`；
- 仍失败时只标为 `full_skill_failure_pending_analysis`，不会自动宣判 skill deficit；
- 原 group 含环境/runtime 技术错误时不触发 skill 分析。

round 结束后复制并修改在线分析示例配置，再运行：

```powershell
Copy-Item configs\trace2skill_online_analysis.example.yaml `
  configs\trace2skill_online_analysis.yaml
python scripts\run_shopsimrl.py training analyze `
  configs\trace2skill_online_analysis.yaml
```

该命令只消费失败的 full-skill retry trajectory，调用现有 live、gold-aware Failure Analyst。ADD 必须有真实成功 counterfactual trial；REWRITE 的 target 必须是 current active chunk；gold firewall 与 cold start 相同；Analyst 仍可返回 `NO_PROPOSAL`。随后所有 eligible cards 被一次 many-to-one compiler 合并，近义机制去重后每轮最多输出 **6 条 ADD/REWRITE candidates**（`max_candidates` 默认值及当前配置均为 6）。单条 trajectory card 不会直接成为 validation factor。第二轮起可在配置中指定上一轮 `proposal_ledger.jsonl`；compiler 只把它作为审计/去重状态，不注入 policy prompt，gate 输出会累计历史并更新本轮候选结果。

主要产物：

- `failure_cards.jsonl`：privileged audit 与 deployable abstraction 分离；
- `candidate_pool.json`：current chunks + validation-ready ADD/REWRITE；
- `proposal_ledger.jsonl`：等待 validation 的候选；
- `analysis_summary.json`：覆盖和候选统计。

## 4. 当前 checkpoint 的 randomized validation

在线 gate 使用固定 `val` 400 tasks，每个 task 一条 bare 加一条 masked rollout，统一拟合 \(R_i-B_i=C_i^\top\beta+\epsilon_i\)。先将当前 checkpoint 以 OpenAI-compatible endpoint 暴露；`configs/qwen35_4b_val_slime.yaml` 默认连接 `http://127.0.0.1:30000/v1`。每轮更新 `experiment.name` 与 `model.checkpoint_id` 为实际冻结权重的唯一标识，并保持服务在 bare/masked/补测期间不换权重。

先用原始 evaluator 取得 bare validation score；已有相同 checkpoint、同协议的完整 run 时直接复用这批 traces，不重复调用模型。`online_gate.bare_run_dir` 指向该 run，`experiment_config` 引用同一份 bare YAML：

```powershell
python scripts\run_shopsimrl.py run configs\qwen35_4b_val_slime.yaml
# 已有匹配的完整 bare run 时跳过上一条，并设置 bare_run_dir。
Copy-Item configs\trace2skill_online_gate.example.yaml `
  configs\trace2skill_online_gate.yaml
python scripts\run_shopsimrl.py training online-gate-plan `
  configs\trace2skill_online_gate.yaml
python scripts\run_shopsimrl.py training online-gate `
  configs\trace2skill_online_gate.yaml
```

普通 current/ADD slot 使用 `absent/present` 两个等概率状态。存在 rewrite family 时，一个逻辑 slot 使用 `absent/old/new1/...` 外生互斥状态；old 和 new 不会同时出现在 prompt。所有版本 dummy columns 与其他 slots 一起进入 **paired-delta 无截距 main-effect OLS**，因变量是同任务的 masked reward 减 bare reward，各 `r_*` 分量同样逐项相减。绝不减全局 bare 均值、不另外估计截距、不复用上一 checkpoint 的基线。

共享数据契约、完整性/provenance 校验和解释边界见 [paired_validation.md](./paired_validation.md)。训练中的 skill-free group 或 full-skill retry 不能替代固定 val 的 bare run；当前是有限 round 后显式调用 evaluator/gate 的流程，未新增 trainer 内部自动评测调度。

选择顺序严格固定：

1. 每个 rewrite family 内按 reward coefficient 选唯一 winner；相等时保留 old；
2. loser 在全局排名前废弃；
3. unchanged、ADD 与 rewrite winner 合成 survivor pool；
4. 只对 coefficient `> 0` 的 survivors 做全局 top-K；
5. current chunk 的非正或 budget-excluded 结果记为 retired；
6. new rewrite winner 进入 active bank 时沿用 target 的逻辑 `skill_id`，避免每轮改写破坏 slot 身份。

原始 bare run 的 `summary.json` 继续作为训练汇报的 skill-free validation score；gate 的 `summary.json` 则是 masked 原始分数。`contributions.json` 单独保存差值、系数和 baseline 来源/hash，不覆盖两组绝对指标。符号用于当前选择规则，不单独证明正面/负面因果作用。

Gate 完成后输出 `mask_assignments.json`、通用 evaluator traces/summary、`contributions.json`、带最终 validation result 的 `proposal_ledger.jsonl`、`selected_skillbank.json`、`selected_skill.md` 和 `online_gate_manifest.json`。任何 coverage 缺失都会保留 `incomplete`，不会冻结 selection。

## 5. 下一轮

用新 gate 的 `selected_skillbank.json` 显式选择下一轮 q/rho 参数并构建新的 curriculum，然后让 slime 从对应 checkpoint 继续一个有限 round。

注意 slime 的 `NUM_ROLLOUT` 是累计结束位置，不是本轮增量。第一轮更新到 100 后若再训练 100 轮，应从已保存 checkpoint 加载并设 `NUM_ROLLOUT=200`；仍填 100 会使循环为空。同时更新两个输入路径、`round_id` 和评测的 checkpoint identity。

完整循环为：

```text
validated SkillBank + explicit q/rho
        → frozen curriculum state
        → slime GRPO round
        → all-wrong full-skill retry queue
        → gold-aware online analysis + many-to-one candidates
        → same-checkpoint bare val score + reusable task traces
        → frozen-checkpoint paired-delta randomized val gate
        → replacement winner + positive top-K
        → next validated SkillBank
```

Test split 不参与上述任何更新。最终 checkpoint 只在 test 上分别报告 full-skill paired performance 与 skill-free performance，二者差值作为 internalization gap。

## 6. 无 GPU 链路回归

```powershell
python -m pytest tests/test_slime_runtime_integration.py tests/test_training.py tests/test_episode_admission.py tests/test_training_preflight.py tests/test_paired_validation.py -q
```

## 7. 训练监控

W&B 的登录、run/group 组织、完整指标口径、Analyst/gate Table 与 Artifact、
offline sync、建议面板和异常排查见 [training_monitoring.md](./training_monitoring.md)。

集成测试直接使用当前 slime 的 OpenAIAdapter、TrajectoryManager 和 Sample，以本地替身提供模型输出与环境，验证多轮 token/logprob/loss mask、工具修复、中止、group reward、diagnostic retry 不进入 trainer、失败 retry 到 online analysis/candidate pool、在线 gate 到下一轮 curriculum。未启动 GPU 更新或真实 Analyst/Compiler API。真实 SGLang parser/服务、环境联调和 Megatron forward/backward 留给少量卡 smoke。
