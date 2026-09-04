# ShopSimRL 代码规划：Agent Runtime 与评测流水线

> 状态：v0.1，首版基础设施已落地
> 对应研究设计：`docs/proposal.md` v0.2
> 当前范围：ShopSimulator single-turn / personalization、随机轨迹采样、离线 trace 分析、Skill 接入和 checkpoint 评测

## 1. 结论与现状判断

`ShopSimulator/shop_env` 已经提供了稳定的环境边界：租约式 session、`observation_state`、显式 action-limit termination、论文奖励分量和稳定 task split。这个边界应继续作为唯一环境真相，不应在实验代码中复制页面状态机、动作解析或奖励逻辑。

应弃用的是 `ShopSimulator/single_eval` 的 runner，而不是环境。旧 runner 的主要问题是：

- 模型 API、Agent 状态机、环境客户端、并发、重试和 JSON 落盘耦合在一个文件；
- 用 `task_nums + range(...)` 推导任务，无法正确表达带空洞的稳定 task ID；
- 每个 task 只保存最终 conversation，缺少逐步模型元数据、canonical observation、无效动作原因、终止原因和采样 provenance；
- 失败被打印后吞掉，不形成可重试、可统计的失败 artifact；
- prompt、persona 和未来 skill 注入没有独立版本边界；
- checkpoint 评测不能保证使用完全相同的冻结任务样本；
- 随机采样脚本和正式评测各自维护一套 rollout 逻辑，后续容易产生口径漂移。

因此新代码放在根目录独立 `shopsimrl` 包中，只通过 HTTP 使用 ShopSimulator，不继续修改当前已调好的环境，也不依赖 `single_eval`。

## 2. 设计目标与非目标

### 2.1 设计目标

1. **一条 rollout pipeline，多种实验用途。** Teacher 随机采样、base/SFT 诊断、val 评测和 checkpoint 对比只改变配置，不复制 runtime。
2. **trace 是核心研究数据。** 一条 episode artifact 应足够支持重放审计、错误分类、SFT 候选筛选、skill 构造和指标重算。
3. **环境、模型、prompt、skill 可替换。** Agent 状态机不感知具体 API 服务商、checkpoint 或 skill 检索算法。
4. **默认可复现、可恢复。** 任务采样、每轨迹 seed、manifest、版本与完整性判断都有稳定规则；中断后只跳过真正完成的 episode。
5. **评测公平。** 每个 checkpoint 单独运行；使用相同 split manifest、seed、采样参数和 prompt hash 即可重建完全相同的 job 列表。
6. **对未来训练友好。** 核心协议保持轻量，后续可以增加 SGLang/slime adapter，而不改变 trace 和评测口径。

### 2.2 当前非目标

- 不在基础设施阶段冻结 skill taxonomy、检索算法、context 模板或 curriculum schedule；
- 不重写 ShopSimulator 的动作 parser、页面状态、search 或 reward；
- 不把 LLM-as-a-Judge 混入在线 Agent loop；Judge 应是消费冻结 trace 的独立离线阶段；
- 不在当前版本实现多轮 Shopper simulator；
- 不让最终 test 结果驱动 prompt、skill 或 checkpoint 选择。项目 split manifest 已冻结；最终 test 的访问保护仍需在正式实验流程中落实。

## 3. 总体架构

```text
frozen task split ──> deterministic jobs ──> Evaluator / resume / concurrency
                                                │
                                                ▼
                                  AgentRuntime (one episode)
                              ┌─────────┼──────────┐
                              ▼         ▼          ▼
                         ChatModel   SkillProvider  ShopEnvironment
                              │         │          │
                              └─────────┼──────────┘
                                        ▼
                               versioned episode trace
                                        │
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
              metric summary      trace diagnosis     SFT / Skill build
```

每个 `AgentRuntime` 只负责一个 episode，并拥有一个 model client 和一个 environment lease。`Evaluator` 只负责 job 调度、并发、断点续跑和落盘，不参与 Agent 决策。这样可以单测状态机，也可以独立替换模型或环境 adapter。

### Bare validation traces 的复用

Cold-start Gate A 与训练 online gate 统一采用 \(R_i-B_i=C_i^\top\beta+\epsilon_i\) 的无截距 OLS。通用 evaluator 在 `skills.path: null` 时产生的完整 bare val run，可通过 gate 的 `bare_run_dir` 复用；其 summary 继续报告原始 skill-free validation score，gate 只增加 masked rollouts，差值只用于回归。

普通模型配置新增可选 `model.checkpoint_id`，进入 provenance/fingerprint 而不发送给 API；训练 online gate 要求显式设置它，以免同一服务地址换权重后误用旧基线。详细配对检查和输出约定见 [paired_validation.md](./paired_validation.md)。

## 4. 目录规划

```text
shopsimrl/
  schemas.py       # EpisodeJob、ModelOutput、Skill 和 schema/version helpers
  model.py         # ChatModel 协议与 OpenAI-compatible HTTP adapter
  environment.py   # ShopEnvironment 协议与 ShopSimulator HTTP adapter
  prompts.py       # persona / skill / observation 的版本化 prompt 组装
  skills.py        # SkillProvider 协议、NoSkills、最小 JSON SkillBank adapter
  runtime.py       # 单 episode Agent 状态机和完整 trace 生产
  tasks.py         # 显式 task ID / legacy range split 加载
  store.py         # atomic JSON、run manifest、trace store 和 resume
  evaluation.py    # 固定采样、并发执行、聚合指标
  paired_validation.py # cold-start/online 共用的 bare 配对校验与逐任务 delta
  config.py        # YAML 实验配置与路径解析
  cli.py           # plan / run / summarize

configs/
  prompts/persona_single_turn.txt
  qwen35_4b_train.yaml
  qwen35_4b_val.yaml
  qwen35_4b_test.yaml

docs/
  code_plan.md

tests/
  test_runtime.py
  test_evaluation.py
  test_tasks_and_skills.py
```

后续模块在证据和需求出现后再增加，建议名称如下：

```text
shopsimrl/analysis/       # 规则标签、错误 taxonomy、trace 查询与导出
shopsimrl/judge/          # Judge rubric、批处理、校准和原始输出
shopsimrl/skill_build/    # candidate 提取、去重、泄漏检测、人工审核
shopsimrl/training/       # slime/SGLang rollout adapter、RL sample 转换
```

这些后续模块只消费已版本化 trace 或实现现有协议，不回写环境。

## 5. 稳定协议

### 5.1 EpisodeJob

一个 job 由四项唯一确定：

```json
{
  "task_id": 1234,
  "sample_id": 0,
  "seed": 287413091,
  "split": "train"
}
```

`episode_id = task-{task_id}__sample-{sample_id}`。`sample_id` 表示同一任务的第几次独立采样，不能用随机 UUID 替代，否则断点续跑和跨 checkpoint 对齐会变得困难。episode seed 由实验 seed、task ID 和 sample ID 稳定哈希得到。

### 5.2 ChatModel

输入是标准 OpenAI message 列表、当前页面动态 function tools 和 episode step seed；输出统一为：

- 标准 `tool_calls`（每轮恰好一个）及解析后的 JSON arguments；
- 服务商单独返回的 thinking `reasoning_content`；
- 可选的最终 `content`，动作执行不再从文本中解析；
- `finish_reason`、token usage、response ID、延迟和少量非敏感元数据。

runtime 根据 `observation_state.search_available/actions` 动态暴露 `search(query)` 和 `click(value)`；click 参数使用当前可点击值的 enum。assistant tool call 原样进入 conversation，环境 observation 以匹配 `tool_call_id` 的 `role=tool` 消息返回。文本 `Thought/Action` 不作为兼容路径。

API key 只从配置指定的环境变量读取，永不进入 manifest 或 trace。当前 HTTP adapter 支持 OpenAI、vLLM、SGLang 和 SiliconFlow 风格 endpoint，并允许通过 `extra_body` 传入 `enable_thinking` 等服务商参数。远程 API 可保留 `trust_env=true` 使用系统代理；本地 vLLM/SGLang endpoint 应设为 `false`，避免 localhost 请求被代理截获。

后续若训练 rollout 不经过 HTTP，只需实现相同 `ChatModel.generate()` 语义，或增加异步等价协议；Agent loop 不需要改写。

### 5.3 ShopEnvironment

环境协议只有：

```text
reset(task_id) -> reset payload
step(canonical_action) -> step payload
terminate(reason) -> terminal payload
close() -> release lease
```

新 adapter 强制检查 canonical `observation_state` 和 observation version，reset 永远传 `include_private_goal=false`。动作解析、是否 done 和 reward 完全服从环境返回；runtime 不添加 `finish`、错误购买惩罚或自定义成功规则。

### 5.4 SkillProvider

`SkillProvider.select(public_episode_context)` 返回零个或多个带 ID、版本、正文和 metadata 的 skill。public context 只包含 task instruction、persona、初始 observation state 和 job 信息，不包含私有 goal。

当前 `JsonSkillBank` 只是用于接口验证和可复现小实验的最小 adapter：支持 global skill 和显式 task-scoped skill。它不是最终 retrieval 设计。最终的 embedding/router/LLM selector 只需实现同一协议，并在 `identity()` 中记录索引、模型、prompt 和 bank snapshot 哈希。

## 6. Trace 规范

同一 run 的轨迹逐条追加到 JSONL：

```text
runs/<experiment>/traces.jsonl
```

`shopsimrl-episode-v4` 至少包含：

- `job`：task、sample、seed、split；
- `status`：`completed` 或 `failed`；
- `provenance`：model、sampling、environment/observation version、prompt hash、SkillBank hash、runtime action limit；
- `reset`：公开任务、persona、初始 observation/state，不保存 lease，也防御性移除私有 goal；
- `selected_skills`：实际注入内容、ID、版本、content hash 和选择 metadata；
- `conversation`：模型真实看到的 system/user/assistant/tool 消息和 tool-call ID；
- `steps`：每步输入 observation、动态 tool schema、thinking、结构化 tool call、转换后的 canonical action、usage/延迟、完整环境结果和 `action_feedback`；
- `final`：reward、reward detail、purchase、goal、termination reason；
- `error`：失败 stage、异常类型、信息和 traceback；
- UTC 时间与 episode 总耗时。

每个 episode 完成后由调度主线程立即 append、flush 和 `fsync`。恢复时逐行读取，以同一 `episode_id` 的最后一条合法记录为准；只有 schema 正确、`status=completed` 且 `final.done=true` 的 episode 才会跳过。失败 episode 会在重跑时追加新记录，损坏或中断的尾行不会被当作完成结果。

Trace 保留模型 reasoning 是为了研究诊断；对外发布、Judge 输入或 SFT 转换时必须走显式 exporter，不能默认把 reasoning、终局私有 goal 或 persona 全量外发。

模型调用的重试边界按责任划分：网络异常、限流和可重试 HTTP 服务错误由模型 adapter 透明重试；HTTP 成功但没有工具调用、调用数量错误、工具名或 arguments 格式错误时，不再重采样。runtime 将原始响应、`protocol_error` 和结构化 `protocol_feedback` 写入当前 step，该 step 消耗一次 action budget，然后在同一页面让模型纠错。能解析为合法函数参数、但 `click.value` 不在动态 enum 中的动作仍交给环境判定并返回标准 `action_feedback`。

协议错误 step 的 `action` 和 `environment` 为 `null`。summary 分开统计 `model_steps`、`protocol_errors` 和实际送入环境的 `total_actions`，避免把模型协议错误混作环境无效动作。

当模型正常返回 `finish_reason=length`，但尚未产生唯一有效工具调用时，这属于有效的模型策略失败而非基础设施错误。runtime 保留 reasoning 和原始响应，记录 `policy_failure.code=generation_length`，立即请求环境以 `generation_length` 终止并生成标准零分 reward。该轨迹进入训练和评测，不重采样，也不追加协议纠错轮次。

## 7. Run manifest 与 artifact 布局

```text
runs/<experiment>/
  manifest.json
  summary.json
  traces.jsonl
```

manifest 冻结所有影响语义的内容：具体 job 列表、episode seed、task split 版本、model 与 sampling 参数、environment mode、prompt hash、skill snapshot hash 和 action limit。`concurrency`、输出位置和 `resume` 是执行参数，不影响语义 fingerprint，因此可在续跑时调整。

若同一 `<experiment>` 的语义 fingerprint 变化，程序拒绝覆盖，要求换新 experiment 名。这可以避免“同一结果目录混入两套 model/prompt/temperature/task”的常见问题。

## 8. 三类主要工作流

### 8.1 Teacher 随机采样

使用项目冻结 manifest 中的 `train`、设置 `sample_size`、`repeats`、较高 temperature 和 Teacher 模型即可。成功与失败 trace 使用完全相同的 schema，采样阶段不按 reward 丢数据。

```powershell
python -m shopsimrl.cli plan configs\teacher_sample.yaml
python -m shopsimrl.cli run configs\teacher_sample.yaml
```

后续 SFT exporter 从这些 trace 中筛选 `completed + r_success=1`，再做语义质量、重复率和覆盖检查；不能让在线 sampler 同时承担筛选逻辑。

### 8.2 Trace 分析与 Skill 构造

分析阶段只读取冻结 trace，建议依次产生：

1. `episode_labels.jsonl`：规则标签、阶段错误、无效动作、检索/规格/预算诊断；
2. `judge_outputs.jsonl`：Judge 原始输出、rubric/version、解析结果；
3. `skill_candidates.jsonl`：候选正文、正反 evidence episode/step refs、生成方法；
4. `skillbank.snapshot.json`：去重、泄漏检测和人工审核后的不可变快照。

每个 skill 至少应可追溯到 evidence refs，并记录 builder model/prompt、去重簇、审核状态和泄漏检查结果。最终 schema 和 taxonomy 等轨迹分析后再冻结；runtime 已经只依赖最小 `Skill` 对象，不会阻塞该研究选择。

### 8.3 单 checkpoint 固定评测

一份 YAML 只描述一个 checkpoint 和一个 split。每得到一个 base、SFT、GRPO 或训练 step checkpoint，就复制配置、修改 `experiment.name` 与 `model` 后单独运行。只要 split manifest、seed、`sample_size`、`repeats` 和 prompt/skill 配置不变，job 列表便完全一致；跨 checkpoint 对比由后续分析读取各自 `summary.json` 或 `traces.jsonl` 完成，不在在线 runner 中生成 comparison artifact。

主要报告：

- coverage 与失败数，防止静默排除 API/环境失败；
- `reward` / `r_strict`、`r_loose`、`r_success` 和所有 `r_*` 分量；
- 平均步数、无效动作率、终止原因；
- token 和 wall-clock 成本；
- reward mean 的 SEM，用于快速判断波动，正式报告再按 task 对齐做 bootstrap/paired test。

最终 test 配置必须 `skills.path=null`。checkpoint 选择只能使用 train/val 结果；选定 checkpoint 和配置 fingerprint 后再单次运行锁定 test。

## 9. 配置原则

`configs/qwen35_4b_{train,val,test}.yaml` 是当前可直接运行的完整配置。其他模型或 checkpoint 从最接近用途的文件复制后修改：

- 随机采样：`split=train`、`sample_size=N`、`repeats=K`、Teacher model、非零 temperature；
- 固定评测：`split=val/test`、固定 temperature、单个 checkpoint、通常 `repeats=1`。

当前 Qwen3.5-4B 配置统一使用 thinking mode 和标准 function tool calling，只以 `max_tokens=2048` 限制单轮生成，不设置独立 thinking budget；temperature 为 0.6、`top_p=0.95`、`top_k=20`。prompt 不要求输出可见 Thought 或文本 Action，thinking 与 tool call 分字段进入 trace。

先运行 `plan`。它只验证 YAML、split、persona 一致性、SkillBank 和确定性抽样，不调用模型或环境 API。确认 episode 数、前几个 task ID、prompt/skill hash 后再运行 `run`。

## 10. 与训练框架的边界

评测 runtime 不应直接依赖 slime 的内部 sample 类型。建议增加双向纯转换层：

```text
shopsimrl episode/job  <->  slime rollout request/sample
shopsimrl trace        <->  SFT/GRPO training record
```

训练 adapter 应继续复用：

- 相同的 prompt builder 和 skill renderer；
- 相同的 ShopEnvironment 语义；
- 相同 episode/step trace 字段；
- 相同 reward 与 termination 口径。

训练框架特有字段（token IDs、loss mask、old logprob、advantages）放在独立 `training` namespace，不污染通用评测 trace。这样未来更换 slime 版本不会迫使我们迁移历史轨迹。

## 11. 迁移方案

1. 删除 `ShopSimulator/single_eval`、`multi_eval` 及其历史输出，不保留兼容入口；
2. 所有后续采样和评测统一使用 YAML pipeline 与 `scripts/run_shopsimrl.py`；
3. 先用 fake runtime 单测，再用环境 smoke endpoint 跑 1–3 个 task；
4. 用环境契约测试固定 task input、动作数、reward detail 和终止原因；
5. 确认一致后生产 Teacher 轨迹，并冻结首个 trace schema；
6. 轨迹分析完成后再实现 `analysis/`、`skill_build/` 和最终 SkillBank selector；
7. 最后增加 slime rollout adapter，避免基础评测和训练框架同时快速变化。

## 12. 当前已实现与下一阶段

本次已经实现：

- 独立的 runtime/model/environment/prompt/skill 协议；
- OpenAI-compatible 模型调用与有限重试；
- canonical ShopSimulator HTTP adapter 和可靠 lease 清理；
- 确定性随机采样、同 task 多轨迹、并发和断点续跑；
- 完整 episode trace、run manifest 和 summary；
- skill-free 与最小 JSON SkillBank 路径；
- `plan / run / summarize` CLI、示例配置和 fake 端到端单测。

项目数据划分已冻结为 `persona_splits.v1.json`：`train=3726`、`val=400`、`test=400`，并按商品、指令、persona 和用户标识的精确关联分组，避免关联样本跨 split。

建议下一阶段按以下顺序推进：

1. 用本地环境和一个可控模型 endpoint 完成 3-task integration smoke test；
2. 在正式实验入口增加 split role/test access guard，避免开发阶段误用最终 test；
3. 为现有 Teacher 轨迹写 converter 或直接重新采样少量数据，验证 trace analysis 字段；
4. 实现规则型 `analysis` 首版和 SFT exporter；
5. base/SFT 诊断后再冻结 skill candidate 与 SkillBank schema；
6. 接入 slime rollout，并用相同 checkpoint 在离线 eval 与训练侧抽样做口径对齐。
