# Trace2Skill 冷启动实现

当前实现把 [trace2skill.md](./trace2skill.md) 的 cold-start 设计落为四个可断点续跑的步骤：

1. 按 `episode_id` 读取 JSONL 中最后一条记录，并严格检查数量、`train` split、完成状态与 `r_success`。
2. 成功轨迹产生 Success Pattern Cards；失败轨迹在独立的实时 ShopSimulator 会话中做 gold-aware audit，产生 Failure Diagnosis Card 或 `NO_PROPOSAL`。
3. 成功/失败 cards 分通道批量聚类、去重，再把两路 cluster summaries 交给 Holistic Initial Skill Compiler。
4. 输出最多 16 个 canonical draft chunks。它们仍是 pre-validation draft，不能作为 active skill；后续 Gate A 在 val 上估计贡献并筛选 active skill，Gate B 再在 test 上评估 equipped agent。

## 运行

先启动 ShopSimulator 服务；失败分析必须使用实时环境，不能只读离线日志猜根因。当前正式配置通过阿里云百炼的 OpenAI-compatible endpoint 调用 `qwen3.5-plus`，Success Analyst、Failure Analyst、Consolidator 和 Compiler 使用同一个模型；API key 从 `.env` 中的 `BAILIAN_API_KEY` 读取：

百炼的 `qwen3.5-plus` 在 thinking mode 下不支持 `tool_choice="required"`，因此配置使用 `tool_choice: auto`。Trace2Skill prompt 仍要求每轮恰好调用一个工具，客户端会校验返回协议与业务字段；一般非交互结构化阶段最多进行三次模型调用（首次调用加两次 repair），面对全部 cluster 的最终 Compiler 最多进行五次。每次 repair 都携带上一版完整参数和精确错误，且只保留最近一版，避免上下文累积。Compiler 使用 SSE 流式调用并在本地组装增量 `reasoning_content` 与 `tool_calls`，以降低长 thinking 请求被网关或代理断开的概率；若服务未返回流式 tool-call ID，客户端会生成仅用于调用关联的稳定本地 ID，不修改函数名和参数。

```powershell
python scripts\run_shopsimrl.py trace2skill plan configs\trace2skill_cold_start.yaml
python scripts\run_shopsimrl.py trace2skill cold-start configs\trace2skill_cold_start.yaml
```

也可以拆开执行，便于先完成价格较高、耗时较长的 trajectory analysis：

```powershell
python scripts\run_shopsimrl.py trace2skill analyze configs\trace2skill_cold_start.yaml
python scripts\run_shopsimrl.py trace2skill compile configs\trace2skill_cold_start.yaml
```

推荐第一次使用拆分命令：先让 `analyze` 跑完并确认 `analysis_summary.json` 中 `error_count` 为 0，再执行 `compile`。`cold-start` 只是依次执行这两个阶段的便捷入口。

当前 `compile` 内部依次执行 consolidation 和 global compile；consolidation batch 可按 fingerprint 断点复用。Gate A 是独立的 randomized-validation 命令：它在 val set 上对全部 16 个 draft chunks 做联合 masking，复用同 checkpoint 的 bare val traces，用 paired-delta 无截距 OLS 估计贡献并执行 positive top-`K` selection。Gate B 的 equipped rollout 不需要专用执行链，直接使用项目原有的 `plan` / `run` 评测接口；两个已完成 run 的对齐检查和统计报告由 `scripts/compare_shopsimrl_runs.py` 离线完成。

`resume: true` 会按 trajectory 内容、分析模型和关键分析参数的 fingerprint 复用已完成分析；consolidation batch 同样只在输入和 compiler 模型均未变化时复用。

## 后续验证与测试

当前实验固定使用 `runs/qwen35-4b-train-0830/trace2skill-cold-start/initial_skill_draft.json` 中的 16 个 chunks，配置中的 active skill budget 为 `K_init=10`。

先完成或复用同 checkpoint 的 bare val，再执行 Gate A，随后用通用 evaluator 运行 equipped test，最后离线比较 equipped 与已有 bare test run：

```powershell
python scripts\run_shopsimrl.py run configs\qwen35_4b_val.yaml
# 已有匹配且完整的 bare val 时跳过上一条，将 bare_run_dir 指向该 run。
python scripts\run_shopsimrl.py trace2skill gate-a-plan configs\archive\trace2skill_gate_a.example.yaml
python scripts\run_shopsimrl.py trace2skill gate-a configs\archive\trace2skill_gate_a.example.yaml

python scripts\run_shopsimrl.py plan configs\qwen35_4b_test_trace2skill_equipped.yaml
python scripts\run_shopsimrl.py run configs\qwen35_4b_test_trace2skill_equipped.yaml

python scripts\compare_shopsimrl_runs.py `
  runs\qwen35-4b-test-0830 `
  runs\qwen35-4b-test-trace2skill-paired-equipped
```

Equipped YAML 依赖 Gate A 已经生成 `selected_skillbank.json`，因此其 `plan` / `run` 不能在 Gate A 之前执行。Gate A 和通用 evaluator 都使用 append-only traces 与 `resume` 语义；Gate A 未达到 100% coverage 时状态为 `incomplete`，不会冻结 selection。

确认 `bare_run_dir` 中的 bare val 已完整后，需要过夜串行时可运行（脚本不另跑 bare）：

```powershell
python scripts\archive\run_trace2skill_overnight.py
```

该临时编排脚本先运行 Gate A；如 manifest 为 `incomplete`，会利用 evaluator 的 `resume` 语义只补跑未完成任务，默认最多再试 5 次（连同首次最多 6 轮）。Gate A 完整成功后，脚本再调用原始 `plan` / `run` 接口执行 equipped test。配置或 manifest 协议错误不会被盲目重试。

### Gate A：val contribution 与 selected skill

- 数据只使用冻结的 400-task val split，模型 checkpoint 固定为生成裸基线时使用的 Qwen3.5-4B；
- 每个 val task 复用一条 bare rollout，再运行一条 masked rollout；对 16 个 chunks 独立采样 `Bernoulli(0.5)` treatment，形成一个 16 位联合 mask；mask seed、task IDs、draft hash、模型和环境版本写入 manifest；
- 使用 paired-delta 无截距 OLS：\(R_i-B_i=\sum_{j=1}^{16}\beta_jC_{i,j}+\varepsilon_i\)。\(B_i\) 是同 task 的 bare reward，不是全局均值。后续选择记号 \(\alpha_j=\hat\beta_j\)，只包含 16 个 chunk main effects；
- 16 个 chunks 无论是否入选都输出 OLS coefficient、rank 和 status；`r_success` 及其他 `r_*` 分量分别减去对应 bare 分量后，作为辅助贡献报告；
- 先过滤 \(\alpha_j\le 0\)，再在剩余 chunks 中按 \(\alpha_j\) 降序取前 10 个。正贡献不足 10 个时不补齐；同分时以稳定的 `chunk_id` 顺序打破平局；
- 选择完成后生成只包含 selected chunks 的 active SkillBank，并冻结 chunk 内容、顺序、`K_init`、估计器和所有运行配置。

Gate A 至少产出 mask assignment、原始 val traces、16-chunk contribution table、selected SkillBank 和可复现 manifest。逐 chunk SkillBank 在后续运行中必须设置 `max_skills: null` 或 `max_skills >= selected chunk count`；当前 Gate A 基础 val 配置和 equipped 配置均已设为 `null`，不得改回 `4`，否则只会注入前四个 chunks。

当前实现将 Gate A 产物写入配置的 `gate_a.output_dir`：

- `mask_assignments.json`：运行前冻结的 task-level 联合 mask、mask seed、draft hash 和 assignment hash；
- `traces.jsonl` / `summary.json` / `manifest.json`：通用 evaluator 的原始结果、聚合结果与语义计划；
- `contributions.json` / `contributions.csv`：paired-delta 无截距 OLS 系数、rank 和 status；JSON 另存 bare 来源/hash、逐任务 raw/bare/delta 与三者均值；
- `selected_skillbank.json` / `selected_skill.md`：按 canonical draft order 注入的 positive top-`K_init` active skill；
- `proposal_ledger.jsonl`：写入 `selected`、`non_positive` 或 `budget_excluded` 结果的 Gate A ledger 快照；
- `gate_a_manifest.json`：冻结输入、mask、估计器、模型/环境配置和产物 hash。

方法及校验细节见 [paired_validation.md](./paired_validation.md)。`gate_a.bare_run_dir` 必填；当前新产物写入 `gate-a-paired`，equipped 配置也使用新 run 名，避免覆盖旧带截距实验。缺失/未完成/错配 baseline 会在模型调用前报错；gate 不会回退到旧估计器。训练阶段使用同一实现，复用该 checkpoint 已报告 bare score 的 traces。

Mask 使用固定 seed 对 `(split, task_id, sample_id, chunk_id)` 做独立伪随机 Bernoulli 分配，因而并发完成顺序和断点续跑不会改变 treatment。分析前还会逐条核验 trace 中实际 `selected_skills` 与冻结 assignment 一致，并要求 `reward == r_strict`。

### Gate B：test 上的 equipped-vs-bare 对比

- 在冻结的 400-task test split 上，每个任务始终注入 Gate A 选出的完整 skill，不再随机 mask，也不按任务检索不同子集；
- 使用 `configs/qwen35_4b_test_trace2skill_equipped.yaml` 通过原始 `run` 接口执行。该配置按既有 bare run 的 manifest 冻结模型、prompt、persona、工具协议、解码参数、`max_steps`、环境与 reward 设置，只改变实验名和 SkillBank 注入；
- 裸模型对照固定为 `runs/qwen35-4b-test-0830`：400/400 tasks 完成，`reward_mean`/`r_strict=0.506125`，`r_success=0.4825`；
- 报告 equipped agent 的绝对指标及其相对裸模型的 task-aligned 差值。主指标为 `reward`/`r_strict` 与 `r_success`，辅指标包括其他 `r_*` 分量、步数、invalid action/protocol error rate 和 token 成本；正式比较使用 task-level paired bootstrap confidence interval；
- Test 只做一次冻结后的泛化评估。不得根据 Gate B 结果改 chunk、改 `K_init` 或回到同一 test split 反复筛选；如果发现除 skill 外的配置不一致，应先修正实验对齐，必要时重跑裸基线。

Equipped run 仍只产生通用 evaluator 的 `manifest.json`、`traces.jsonl` 和 `summary.json`。完成后运行 `scripts/compare_shopsimrl_runs.py`：脚本会先逐项对齐两个 run 的模型和解码参数、task jobs/seeds、split、环境、prompt、`max_steps`、action protocol 与 task-level runtime provenance，并核验 bare trace 无 skill、equipped 的每个 task 都看到同一份完整 frozen skill。任一非 skill 差异都会中止比较。脚本把绝对 summary、task-aligned `equipped - bare` 差值和 percentile paired-bootstrap confidence interval 写入 equipped run 下的 `comparison.json` / `comparison.md`。

### Failure audit trial 语义

ShopSimulator 的 `buy now` 会终止当前购物 episode，但不会终止整个 Failure Analyst。Audit runner 会把该次购买的 `reward`、`reward_detail`、`purchase` 和终止原因作为结构化 tool result 返回给 Analyst，然后为同一 task 自动创建一个新的 investigation trial。Analyst 因此可以继续搜索和检查其他商品，并进行多次 counterfactual purchase。

当前 `max_failure_steps: 20` 限制的是每条失败轨迹的 LLM 决策回合总数；最后一回合只允许提交 diagnosis，所以最多有 19 次 `search`、`click` 或格式修复。一次 ADD proposal 必须至少有一个真实 terminal trial 达到 `r_success=1`，否则会被程序强制降级为 `NO_PROPOSAL`。

## 产物

输出目录包含：

- `trajectory_analyses.jsonl`：每条轨迹的一次完整、可恢复分析记录；
- `evidence_cards.jsonl`：从最新 trajectory analyses 精确物化的 cards；
- `analysis_errors.jsonl` / `analysis_summary.json`：失败记录和覆盖统计；
- `cluster_batches.jsonl`：成功/失败分通道 consolidation 结果；
- `initial_skill_draft.json`：Gate A 的结构化 canonical chunk 输入；
- `initial_skill_draft.md`：完整 draft skill 的可读版本；
- `initial_skillbank.json`：与当前 `JsonSkillBank` 兼容的逐 chunk draft；
- `compiler_invalid_attempts.jsonl`：最终 Compiler 每次未通过校验的参数与精确错误；
- `proposal_ledger.jsonl`：pre-validation ADD ledger；
- `trace2skill_manifest.json`：输入、模型、预算与产物统计。

`initial_skillbank.json` 中的每个 record 都标记为 `draft_only: true`，manifest 中 Gate A/B 状态也都是 `not_run`。Gate A 结束后应另行生成 selected active SkillBank；不能原地把 draft bank 当作 active skill。Gate B 状态只记录 held-out test 是否完成，不参与 selected skill 的接纳或淘汰。

Failure card 还会保存完整的 `audit_transcript` 与每次 `terminal_trials`，包括模型工具调用、环境 observation/state、购买结果和奖励，便于复核 Analyst 是否真的验证了其修复。

## Gold firewall

失败分析的 `privileged_audit` 可以保存 gold ASIN、错误选择及复盘证据；进入 consolidation 的只有 `deployable_abstraction`。程序还会拦截 ASIN、长产品标识、task ID 和已知实例标题。触发 firewall 的 proposal 会强制降级为 `NO_PROPOSAL`，不会进入 candidate pool。
