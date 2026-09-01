# ShopSimulator Trace2Skill 方案

> - 状态：已决策，作为后续实现依据
> - 适用范围：初始 shopping skill 构造与训练期 skill candidate generation
> - 上位设计：[proposal.md](./proposal.md)
> - 核心原则：**轨迹分析模块只负责提出候选，当前模型上的 randomized validation gate 负责决定 skill 内容。**

本文档锁定 ShopSimulator 项目中的 Trace2Skill 方案。若与 `proposal.md` 中较早的候选生成描述存在差异，以本文档为准，尤其包括以下三点：

1. 训练中的 all-wrong group 必须先进行一次 full-skill retry；只有该 retry 仍失败，才把 **full-skill retry trajectory** 交给失败分析 agent。
2. Skill edit proposal 只允许 `ADD` 和 `REWRITE`。不显式生成 `DELETE` proposal；现有 chunk 的淘汰由 randomized validation 后的 positive top-\(K\) contribution ranking 自动完成。
3. **Trajectory card 不是 chunk candidate。** Cold start 中的大量轨迹必须先经过 many-to-one consolidation，只有少量、去重且语义可分的 canonical chunks 可以进入 validation gate。

## 1. 模块定位

Trace2Skill 不是本项目的核心创新，也不负责最终决定 skill 应包含什么。它在整个 co-evolution 系统中只承担两个职责：

- 从成功轨迹中提炼可迁移的有效工作流；
- 从经过筛选和复核的失败轨迹中诊断可验证的 skill coverage gap，并提出 `ADD` 或 `REWRITE` candidates。

候选不会直接写入 active skill。所有当前 chunks 和新 candidates 最终仍需在冻结 checkpoint、专用 skill-validation split 和随机 chunk interventions 下统一评估。

因此，本文选择一套足够可靠但不过度复杂的方案：

- 成功与失败轨迹采用非对称处理；
- 成功侧使用轻量、非交互的 Success Analyst；
- 失败侧使用能够访问 gold ASIN 和 ShopSimulator 环境的 agentic Failure Analyst；
- 采用并行的 trajectory-to-card 分析与批量合并，不进行按轨迹顺序的连续 skill editing；
- 不引入完整 Wiki、可训练 skill generator、额外 memory bank 或复杂 multi-agent orchestration。

## 2. 两个使用阶段

### 2.1 Cold start：构造初始 shopping skill

初始轨迹按任务最终结果分成两个集合：

\[
\mathcal T^+
=
\{\tau_i:\operatorname{success}(\tau_i)=1\},
\qquad
\mathcal T^-
=
\{\tau_i:\operatorname{success}(\tau_i)=0\}.
\]

两类轨迹分别处理：

- \(\mathcal T^+\) 交给 Success Analyst，提炼成功模式；
- \(\mathcal T^-\) 交给 gold-aware Failure Analyst，进行环境复盘和根因验证。

两路 analyst 的直接输出都是 evidence cards，而不是可直接送入 validation 的 candidates。即使约 400 条初始轨迹产生了数百条 cards，也不能把它们分别作为回归变量；这样既会造成统计功效不足，也会使近义规则因为功能替代而互相稀释贡献。

Cold start 必须采用以下 many-to-one compilation：

```text
约 400 条 initial trajectories
                │
                ├── successes → Success Pattern Cards
                │
                └── failures  → Failure Diagnosis Cards
                                  （gold-aware replay）
                │
                ▼
        成功、失败两侧分别聚类与去重
                │
                ▼
       按相同能力机制对齐两侧脱敏证据
                │
                ▼
          Holistic Initial Skill Compiler
                │
                ├── 合并近义规则
                ├── 解决冲突
                ├── 删除实例特定经验
                ├── 合并不可分的强耦合步骤
                ├── 把长尾证据留在 evidence ledger
                └── 强制满足 regression-factor cap
                │
                ▼
       少量、规范化、语义可分的 canonical chunks
                │
                ├── Gate A：val 上联合 randomized masking 与 positive top-K
                └── Gate B：冻结 selected skill，在 test 上对比裸模型
```

#### 2.1.1 Card-level consolidation

成功与失败证据先在各自通道内归并：

- Success cards 按促成成功的机制、适用条件和工作流阶段聚类；
- Failure cards 按经过复盘验证的最早失败机制、适用条件和建议修复聚类。

完成各自去重后，再在机制层对齐两类证据。例如，“成功轨迹在切换目标 variant 后复核价格”与“失败轨迹使用默认价格而超出预算”共同支持一个 `Variant and price verification` 规则，而不是两个近义 candidates。

一个机制若同时具有 success evidence 和 failure evidence，可以获得更高的 compilation priority；但 failure-only 的稀有机制只要经过环境复盘验证、影响严重且具有明确可观察条件，也可以保留。这里的 priority 只用于压缩和排序，不是 causal contribution。

#### 2.1.2 Initial skill skeleton 与 chunk budget

Initial Skill Compiler 使用购物工作流 skeleton 作为组织先验，而不是把每个 trajectory patch 直接拼接进文档：

```text
Shopping Skill
├── Query formulation
├── Explicit constraints
├── Persona-grounded preference reasoning
├── Candidate inspection and comparison
├── Variant and price verification
└── Backtracking and purchase termination
```

Skeleton 只帮助组织证据，不禁止数据中发现新的机制。每个模块通常保留一到两个 intervention-level atomic chunks；必须共同执行才能生效的短流程应被编译成一个复合 chunk，而不是拆成多个高度相关的回归变量。

第一版实现区分 validation dimension cap 与 active skill budget：

- **Validation dimension cap：** Initial Skill Compiler 最多输出 \(M_{\mathrm{init}}\le 16\) 个 canonical candidate factors；
- **Active skill budget：** validation 后最多保留 \(K_{\mathrm{init}}\) 个 chunks，第一版建议 \(K_{\mathrm{init}}\in[8,12]\)，并要求 \(K_{\mathrm{init}}\le M_{\mathrm{init}}\)；
- 若 compiler 仍产生超过 16 个机制，必须继续合并、提升抽象层级或把低优先级 cards 留在 evidence ledger，不能把超额 candidates 全部送入 validation。

该范围是面向当前购物任务和 validation budget 的工程约束，不是方法的理论常数。后续如果 rollout budget 或任务族复杂度发生明显变化，可以重新设定，但始终应满足：candidate dimension 显著小于有效 validation observations，并为 policy sampling variance 留出余量。

#### 2.1.3 Validation 前的 train-only compilation criteria

Initial Skill Compiler 只能使用 train trajectories、脱敏 cards 和当前/seed skill，不得查询 skill-validation reward。它根据以下证据决定合并、保留或暂存：

- 是否在多个独立任务上重复出现；
- failure diagnosis 是否通过环境重放和最小修复验证；
- 是否同时得到成功和失败证据支持；
- 触发条件是否能从 task、persona 或正常 observation 获得；
- 是否跨商品 category 泛化；
- 是否与其他规则重复、冲突或存在强耦合；
- 相对于上下文成本，是否覆盖了新的能力机制。

这个步骤只负责把大量文本假设编译成少量、可辨识的 intervention factors，不声称估计 chunk 的真实回报贡献。因此它不会取代 randomized validation，也不会消耗 skill-validation split 来反复搜索文本方案。

#### 2.1.4 Cold-start Gate A/B：val 筛选，再做 held-out test

Compiler 先产出 initial skill draft

\[
S_0^{\mathrm{draft}}=\{c_1,\ldots,c_{M_{\mathrm{init}}}\},
\qquad M_{\mathrm{init}}\le 16.
\]

**Gate A 是唯一的 cold-start selection gate。** 在固定 val set 上冻结待测 Qwen3.5-4B checkpoint，对全部 canonical chunks 直接做联合 randomized masking：

\[
C_{i,j}\sim\operatorname{Bernoulli}(0.5),
\qquad
S_i=\{c_j:C_{i,j}=1\}.
\]

400 个 val tasks 各运行一次。评估器按固定 seed 为每个 task 独立生成一个 16 位联合 mask，共得到 400 条 treatment-reward observations。Chunk contribution 使用带截距的 naive OLS 估计：

\[
R_i
=
\beta_0
+
\sum_{j=1}^{M_{\mathrm{init}}}\alpha_{0,j}C_{i,j}
+
\varepsilon_i.
\]

16 个 chunks 是彼此独立的最小原子干预单元，回归只包含 main effects。各列 \(C_{i,j}\) 完全外生，任务间难度差异进入 \(\varepsilon_i\)。\(\alpha_{0,j}\) 表示其他 chunks 按联合 masking distribution 随机出现时，chunk \(j\) 对 reward 的平均边际贡献。主排序指标使用项目主 reward（当前等同 `r_strict`），`r_success` 与其他 reward components 作为辅助结果一并报告。Validation 只接收 canonical chunks。

Gate A 使用统一的 positive top-\(K\) 规则。先取所有 OLS coefficient 为正的 chunks：

\[
\mathcal C_0^+
=
\{c_j:\alpha_{0,j}>0\},
\]

再按 \(\alpha_{0,j}\) 从高到低排序，选取不超过 active skill budget 的前 \(K_{\mathrm{init}}\) 个：

\[
S_0
=
\operatorname{TopK}_{K_{\mathrm{init}}}
\left(\mathcal C_0^+;\,\alpha_{0,j}\right).
\]

因此每个 validation-ready chunk 只有三种确定结果：

- **Selected：** \(\alpha_{0,j}>0\) 且位于 top-\(K_{\mathrm{init}}\)，进入初始 active skill；
- **Non-positive：** \(\alpha_{0,j}\le 0\)，不进入 active skill；
- **Budget-excluded：** \(\alpha_{0,j}>0\)，但排名超出 top-\(K_{\mathrm{init}}\)，不进入 active skill。

如果正贡献 chunks 少于 \(K_{\mathrm{init}}\)，只保留实际为正的部分，不使用零贡献或负贡献 chunk 填满 budget。

Gate A 必须保存全部 \(M_{\mathrm{init}}\) 个贡献估计、排名和 selection status；未入选 chunk 仍保留在 evidence ledger，但不得因查看结果而围绕同一个 val set 无界改写、改 \(K_{\mathrm{init}}\) 或重测。Gate A 输出的 \(S_0\) 及其内容顺序随 val 结果一起冻结。

**Gate B 是 held-out test evaluation，不再参与选择。** 在 test set 上不给 chunk 施加 mask，而是让 agent 的每个任务都看到 Gate A 冻结的完整 \(S_0\)。待测 checkpoint、system prompt、persona、工具协议、解码参数、`max_steps`、环境和 reward 版本必须与既有裸 Qwen3.5-4B test run 相同，实验条件只允许在“注入 \(S_0\)”这一项上不同。若使用逐 chunk `JsonSkillBank`，`max_skills` 必须为 `null` 或至少等于 \(|S_0|\)，避免运行时静默截断 selected chunks。

Gate B 报告 equipped agent 的绝对指标，以及相对裸模型的 task-aligned 差值；主要比较 `reward`/`r_strict` 与 `r_success`，同时报告各 reward component、行为错误率、步数和 token 成本。Test 结果只用于最终泛化报告，不能反向调整 chunk、贡献估计、排序规则或 \(K_{\mathrm{init}}\)。如果 test 配置与既有裸模型 run 无法做到除 skill 外完全一致，应在同一配置下重跑裸模型，而不是直接比较不可对齐的结果。

若 cold start 从空 skill 开始，最终 canonical chunks 在操作语义上都是 `ADD`；若存在 seed skill，则其中也可以包含与 old version 互斥验证的 `REWRITE`。

### 2.2 Online evolution：只分析 full-skill retry failure

在线训练中，以一个 task/prompt 对应的 GRPO group 为单位。设该 group 有 \(G\) 条 rollout：

\[
\{\tau_{i,1},\ldots,\tau_{i,G}\}.
\]

第一版实现使用严格的 all-wrong trigger：

\[
\forall g\in\{1,\ldots,G\},
\quad
\operatorname{success}(\tau_{i,g})=0.
\]

这里的 `success` 使用 ShopSimulator 的最终正确性标准。连续 reward 和 reward breakdown 仍需保存，供后续分析使用，但不改变 all-wrong trigger 的定义。

触发后的固定流程为：

```text
training GRPO group
        │
        ├── 至少一条成功
        │       └── 正常 policy learning；不触发 failure Trace2Skill
        │
        └── 全部失败（all-wrong）
                │
                ▼
         使用完整 current skill 重试一次
                │
                ├── retry 成功
                │       └── 视为 model/internalization deficit
                │           不调用 Failure Analyst，不生成 candidate
                │
                └── retry 仍失败
                        │
                        ▼
             把 full-skill retry trajectory
             交给 gold-aware Failure Analyst
                        │
                        ├── PROPOSE_ADD
                        ├── PROPOSE_REWRITE
                        └── NO_PROPOSAL
```

Full-skill retry 的定义为：

\[
\tau_i^{\mathrm{full}}
\sim
\pi_{\theta_t}(\cdot\mid x_i,S_t),
\]

其中 \(S_t\) 是当前完整 active skill，不施加 chunk mask。

这次 retry 是 **skill evolution 的诊断 rollout**。它不与原来的 masked-skill group 混合计算 GRPO advantage，因为二者 prompt context 不同。是否在其他模块中把 retry 作为 replay data 使用，不属于本文档定义的 Trace2Skill 流程。

需要特别注意：full-skill retry 仍失败，只表示该任务值得进入更深入的 skill-gap investigation，并不自动证明一定缺少 skill。Failure Analyst 仍可以判断为模型执行问题、环境问题或无法可靠归因，并返回 `NO_PROPOSAL`。

## 3. Success Analyst

Success Analyst 采用一次非交互 LLM 分析，不需要 gold ASIN，也不需要环境工具。输入包括：

- task instruction 与 persona；
- 完整成功 trajectory；
- reward breakdown；
- 当前 skill（若存在）。

它不应机械复述整条轨迹，而应识别真正促成成功的决策：

1. 哪个搜索、检查、比较、回退或终止行为是关键；
2. 该行为依赖哪些正常可观察的条件；
3. 哪些动作只是偶然出现或冗余步骤；
4. 当前 skill 是否已经覆盖该策略；
5. 能否形成跨商品 category 复用的工作流。

每条轨迹输出零个或多个 `Success Pattern Card`：

```yaml
source_trajectory_id: ...
mechanism: ...
applicable_when: ...
observed_evidence: ...
decisive_behavior: ...
generalizable_lesson: ...
related_chunk_ids: [...]
confidence: ...
```

成功不等于轨迹中的每个行为都值得保留。若无法识别非平凡、可迁移的有效机制，允许输出空结果。

## 4. Gold-aware Failure Analyst

### 4.1 输入

在线阶段的 Failure Analyst 接收：

- task instruction 与 persona；
- 失败的 full-skill retry trajectory \(\tau_i^{\mathrm{full}}\)；
- 当前完整 skill \(S_t\)；
- 最终选择的商品、variant 和价格信息；
- reward 及其分项；
- gold ASIN；
- ShopSimulator 的正常环境工具与只用于取证的 gold inspection 能力。

原始 masked-skill all-wrong group 只负责触发 full-skill retry。Failure Analyst 的 canonical trajectory input 是随后产生的 full-skill retry failure，避免把“某个关键 chunk 恰好被 mask”误诊成新的 skill gap。

### 4.2 强制分析流程

Failure Analyst 必须采用可验证的环境复盘，而不是只读日志后猜测原因。

#### Step 1：确认 failure surface

结合 reward breakdown，比较 agent 最终选择与 gold 商品，明确失败落在哪一层：

- 搜索召回；
- 显式硬约束；
- persona 偏好；
- 商品详情或相似候选区分；
- variant / option；
- 变体价格与预算；
- 回退、确认或购买终止；
- 环境、奖励或数据异常。

Gold ASIN 是诊断线索，不意味着分析 agent 可以把“选择该 ASIN”本身当作策略。

#### Step 2：重放并定位最早分叉点

重置环境，尽可能复现 full-skill retry trajectory 的 action prefix。分析的目标是找到 **最早的因果性分叉点**，而不是最后一个表面错误。

例如：

- 搜索式过窄，导致 gold 商品从未进入候选；
- gold 商品已经出现，但 agent 因错误理解 persona 而提前排除；
- 商品本体正确，但未切换正确 variant；
- 使用默认展示价格，而没有复核目标 variant 的价格；
- 在仍有未核验约束时过早购买。

#### Step 3：使用 gold 做 oracle-assisted investigation

Failure Analyst 可以在环境中检查 gold 商品，并与 agent 选择的商品逐项比较，以确认：

- 哪条可观察证据被遗漏或误读；
- 正确商品为什么满足任务与 persona；
- 错误商品在哪个约束上不满足；
- 当前 skill 是否已经明确给出了正确处理方式。

#### Step 4：构造并验证最小修复

从最早分叉点出发，仅修改必要的决策，形成一个最小 counterfactual repair。随后在环境中重新执行相关步骤，确认该修复确实能够避免原失败。

如果无法通过环境操作验证根因，不能提出 skill candidate。

#### Step 5：给出诊断与 proposal status

Failure Analyst 最终只能返回以下三种状态：

| 状态 | 含义 | 后续处理 |
|---|---|---|
| `PROPOSE_ADD` | 当前 skill 缺少一个可迁移、可观察、可执行的策略 | 生成 ADD candidate |
| `PROPOSE_REWRITE` | 当前某个 chunk 方向相关，但适用条件、步骤、措辞或核验机制不足 | 生成 REWRITE candidate |
| `NO_PROPOSAL` | 现有 skill 已充分覆盖、问题主要在模型执行；或属于环境/数据异常；或根因无法验证 | 不进入 candidate pool |

`NO_PROPOSAL` 是分析结论，不是第三种 edit operation。

### 4.3 Failure Diagnosis Card

Failure Analyst 输出必须区分 privileged audit 与 deployable abstraction：

```yaml
source_trajectory_id: ...
status: PROPOSE_ADD | PROPOSE_REWRITE | NO_PROPOSAL

privileged_audit:
  gold_asin: ...
  chosen_asin: ...
  failure_surface: ...
  earliest_divergence: ...
  oracle_comparison: ...
  replay_evidence: ...
  minimal_repair: ...
  repair_validation_result: ...

deployable_abstraction:
  failure_mechanism: ...
  applicable_when: ...
  observable_trigger: ...
  corrective_procedure: ...
  verification_step: ...
  target_chunk_id: ... | null
  proposed_content: ... | null
```

只有 `deployable_abstraction` 可以进入后续 consolidation。`privileged_audit` 只用于审计和调试。

## 5. Gold Information Firewall

Gold ASIN 属于 privileged supervision，必须与 policy 和最终 skill 隔离。

### 5.1 可以使用 gold 的地方

- Failure Analyst 对照 gold 与错误商品；
- 在环境中检查 gold 商品的公开页面、属性、variant 和价格；
- 验证哪个决策导致 agent 偏离正确结果；
- 验证最小修复是否针对真实失败机制。

### 5.2 禁止 gold 进入的地方

- policy rollout prompt；
- GRPO update；
- Success Pattern Card；
- deployable candidate chunk；
- candidate merger 的输入；
- 最终 skill；
- 推理时上下文。

进入 candidate pool 前必须执行脱敏和可观察性检查：

- 删除 ASIN、task ID、具体商品标题和实例专属标识；
- 不允许把 gold 独有、正常 agent 无法观察的信息写成条件；
- 每个触发条件必须能从 task、persona 或正常环境 observation 中获得；
- 规则必须描述程序性策略，不能描述该实例的目标答案。

Gold-aware 分析只能提高 candidate proposal 的因果可靠性，不能替代跨任务 validation。

## 6. Candidate Operations

### 6.1 ADD

当现有 skill 没有覆盖一个经过验证的通用失败机制时，提出一个新 chunk：

```yaml
operation: ADD
candidate_id: ...
applicable_when: ...
procedure: ...
verification: ...
common_failure: ...
evidence_card_ids: [...]
```

ADD candidate 必须具有明确适用条件，避免把局部经验写成对所有购物任务都生效的强规则。

### 6.2 REWRITE

当某个已有 chunk 与失败机制相关，但内容不完整、容易误解或不适合当前 policy 时，提出完整替代版本：

```yaml
operation: REWRITE
candidate_id: ...
target_chunk_id: ...
reason: ...
replacement_content: ...
evidence_card_ids: [...]
```

Validation 时 old/new 版本必须互斥，不能同时出现在同一个 prompt 中。为使它们的贡献可以直接比较，把同一逻辑 slot 的 intervention 设计为三个互斥状态：

\[
V_j\in\{\varnothing,\mathrm{old},\mathrm{new}\}.
\]

Old/new 分别相对于“不提供这个逻辑 slot”的共同基线估计贡献：

\[
\alpha_{t,j}^{\mathrm{old}}
=
\mathbb E[R\mid do(V_j=\mathrm{old})]
-
\mathbb E[R\mid do(V_j=\varnothing)],
\]

\[
\alpha_{t,j}^{\mathrm{new}}
=
\mathbb E[R\mid do(V_j=\mathrm{new})]
-
\mathbb E[R\mid do(V_j=\varnothing)].
\]

因此 Rewrite effect 等价于：

\[
\Delta_{\mathrm{rewrite}}
=
\alpha_{t,j}^{\mathrm{new}}
-
\alpha_{t,j}^{\mathrm{old}}.
\]

每个 rewrite pair 必须先在 pair 内完成一次 winner selection：

\[
c_{t,j}^{\star}
=
\arg\max_{v\in\{\mathrm{old},\mathrm{new}\}}
\alpha_{t,j}^{v}.
\]

- 若 \(\alpha_{t,j}^{\mathrm{new}}>\alpha_{t,j}^{\mathrm{old}}\)，new 胜出，old 立即废弃；
- 若 \(\alpha_{t,j}^{\mathrm{old}}\ge\alpha_{t,j}^{\mathrm{new}}\)，old 胜出，rewrite candidate 立即废弃；相等时默认保留 old，避免无收益替换。

Pair 内的 loser 不再参与任何全局排名。只有 winner \(c_{t,j}^{\star}\) 携带自己的贡献值进入 positive top-\(K_t\) selection，并且至多占用一个 active skill budget slot。若同一个 old chunk 在一轮中存在多个 rewrite candidates，则把它们视为同一个 replacement family，在 `old/new₁/…/newₗ` 中只保留贡献最高的一个版本。

### 6.3 为什么不需要 DELETE proposal

当前 chunk \(c_j\) 已经在 validation gate 中接受随机 present/absent intervention，其边际贡献为：

\[
\alpha_{t,j}
=
\mathbb E[R\mid do(C_j=1)]
-
\mathbb E[R\mid do(C_j=0)].
\]

因此，显式 `DELETE` proposal 与 validation gate 的职责重复：

- \(\alpha_{t,j}<0\)：chunk 平均有害，应自动退休；
- \(\alpha_{t,j}\approx 0\)：chunk 已内化、冗余或无效，应降低曝光并退出 active skill；
- \(\alpha_{t,j}>0\)：chunk 对当前 checkpoint 仍有帮助，应继续保留。

即使 Failure Analyst 怀疑某个 chunk 有害，也不提出 DELETE。它可以在 privileged audit 中记录怀疑，由 randomized validation 根据真实 present/absent effect 决定是否删除。

同理，`MERGE` 和 `SPLIT` 不作为外部 operation 暴露：必要的结构调整通过一个 `REWRITE` 加零个或多个 `ADD` 表达，最终仍由 validation gate 决定。

## 7. Consolidation

成功和失败轨迹在分析阶段保持分离，但其脱敏 cards 可以在 consolidation 阶段共同作为证据。

推荐流程：

1. 所有轨迹独立、并行地产生 cards；
2. 成功与失败 cards 先在各自通道中按 mechanism、applicability 和 workflow stage 聚类；
3. 完成通道内去重后，再对齐同一机制的成功证据与失败证据；
4. 去除重复、实例特定和互相冲突的内容；
5. 每个机制产生至多一个规范化 ADD 或 REWRITE candidate；
6. candidates 进入 randomized skill-validation gate。

一个规则若同时得到“成功轨迹中执行该行为”和“失败轨迹中遗漏该行为”的支持，可以获得更高 proposal confidence，但 confidence 只用于排序和审计，不能代替 validation contribution。

默认采用一次 batch consolidation。只有 cards 数量超过上下文容量时，才按机制分桶并做层次化 merge。禁止按照轨迹到达顺序直接、连续地修改 active skill，避免顺序依赖和重复累积。

Cold start 与 online evolution 的 consolidation budget 不同：

- **Cold start：** 允许处理数百张 cards，但必须编译到最多 16 个 canonical validation factors；validation 后再按正贡献 top-\(K_{\mathrm{init}}\) 选择 active chunks；
- **Online evolution：** 每轮只处理 full-skill retry failures 形成的稀疏增量，默认每个机制最多提出一个 candidate，并控制单轮新增/改写规模。

因此，`card`、`cluster-level draft rule` 与 `validation-ready candidate` 是三个不同层级：

```text
card = 一条轨迹提供的证据
cluster-level draft = 多条证据支持的机制总结
validation-ready candidate = 经过 budgeted consolidation 的规范化 chunk intervention
```

## 8. 与 Validation Gate 的接口

Cold-start compiler 输出：

```text
initial_skill_draft
    ├── 最多 16 个 canonical validation factors
    └── 未进入 validation 的 evidence-only cards / cluster summaries
```

Cold-start evaluation 按固定顺序执行：

1. Gate A 在固定 val set 上让 400 个 tasks 各运行一次，对最多 16 个 canonical chunks 做联合 randomized masking；
2. 使用带截距的 naive OLS 估计全部 chunk contributions，丢弃 \(\alpha\le 0\) 的 chunks，对 \(\alpha>0\) 的 chunks 按贡献降序取 top-\(K_{\mathrm{init}}\)，并冻结 selected skill；
3. Gate B 在 held-out test set 上始终注入完整 selected skill，与同口径裸 Qwen3.5-4B 结果比较；test 不参与 selection。

Online Trace2Skill 输出：

```text
current chunks
    + validation-ready ADD candidates
    + mutually-exclusive old/new REWRITE candidates
```

Validation gate 负责：

- 估计所有当前 chunks 的 checkpoint-relative contribution；
- 对每个 REWRITE replacement pair/family 先选择贡献最高的唯一 winner，并立即废弃其余版本；
- 将 rewrite winners 与 unchanged chunks、ADD candidates 合成 survivor pool；
- 对 survivor pool 应用 positive top-\(K_t\) 规则，其中 \(K_t\) 是该轮 active skill budget；
- 决定 ADD candidate 是否进入 active skill；
- 自动退休贡献非正或排名超出 top-\(K_t\) 的现有 chunks；
- 更新下一轮 \(\rho_{t,j}\)；
- 记录 selected / rewrite-loser / non-positive / budget-excluded / retired 结果。

设 \(\mathcal U_t\) 是未被 rewrite 的当前 chunks，\(\mathcal A_t\) 是 ADD candidates，\(\mathcal W_t\) 是每个 replacement pair/family 的唯一 winner。先形成 survivor pool：

\[
\mathcal P_t
=
\mathcal U_t\cup\mathcal A_t\cup\mathcal W_t.
\]

再执行统一的 positive top-\(K_t\) selection：

\[
\mathcal C_t^+
=
\{c_j\in\mathcal P_t:\alpha_{t,j}>0\},
\qquad
S_{t+1}
=
\operatorname{TopK}_{K_t}
\left(\mathcal C_t^+;\,\alpha_{t,j}\right).
\]

这意味着 pair 内比较先于全局排名：loser 即使贡献为正也不会占用 budget；winner 即使战胜另一版本，仍必须满足 \(\alpha>0\) 且进入全局 top-\(K_t\)，才能出现在下一轮 active skill 中。

Trace2Skill 不负责：

- 直接修改 active skill；
- 根据来源轨迹的单次修复结果决定接纳；
- 计算 chunk contribution；
- 控制 \(q_t\) 或 \(\rho_{t,j}\)；
- 把 gold-derived facts 注入模型训练。

## 9. 最小持久状态

不建立完整 Wiki，只维护一个轻量 proposal ledger：

```yaml
candidate_id: ...
operation: ADD | REWRITE
target_chunk_id: ... | null
source_card_ids: [...]
failure_mechanism: ...
proposal_checkpoint: ...
stage: cold_start | online
consolidation_cluster_id: ...
replacement_family_id: ... | null
validation_result: selected | rewrite_loser | non_positive | budget_excluded | retired
estimated_effect: ...
contribution_rank: ...
notes: ...
```

该 ledger 用于：

- 避免重复提出已经被拒绝的同义 candidate；
- 追踪同一失败机制是否反复出现；
- 保存未进入 validation 的长尾证据，以及因 top-\(K\) budget 未被选择的正贡献 candidate；
- 区分“candidate 质量差”和“过去有效、后来被模型内化”；
- 保留 skill evolution 的可审计历史。

Ledger 不进入 policy prompt，也不是推理时 memory。

## 10. 实现不变量

后续代码实现必须满足：

1. Masked-skill all-wrong group 不能直接进入 Failure Analyst；必须先做一次 full-skill retry。
2. Failure Analyst 分析的是失败的 full-skill retry trajectory。
3. Full-skill retry 与原 masked group 不混合计算 group-relative advantage。
4. Failure Analyst 可以返回 `NO_PROPOSAL`；retry 失败不等于必然需要扩张 skill。
5. Gold ASIN 只能存在于 privileged audit path。
6. Candidate operation 只有 `ADD` 和 `REWRITE`。
7. 所有 candidates 只进入 candidate pool，不能直接修改 active skill。
8. 现有 chunk 的删除只由 randomized contribution gate 触发。
9. REWRITE validation 中 old/new 必须互斥。
10. 成功与失败轨迹先分别分析，再在脱敏 card 层合并。
11. Per-trajectory card 不能直接作为 validation regression factor。
12. Cold start 必须先完成 train-only many-to-one consolidation，再进入 validation。
13. 第一版 initial compiler 不得输出超过 16 个 validation factors，active skill budget \(K_{\mathrm{init}}\) 建议设为 8--12。
14. Cold-start initialization 在 val set 上让每个 task 运行一次，并对全部 canonical chunks 做联合 randomized masking。
15. Gate A 的 val 结果是 cold-start chunk 与 \(K_{\mathrm{init}}\) 的最后一次选择依据；Gate B 的 test 结果不得反馈到 skill selection。
16. Cold-start Gate A 使用带截距的 naive OLS，只拟合 chunk main effects。
17. Validation 统一按贡献降序选取 \(\alpha>0\) 的 top-\(K_t\)，不足 \(K_t\) 时不以非正贡献 chunk 补齐。
18. 每个 REWRITE pair/family 必须先保留贡献最高的唯一版本，其他版本在全局排名前立即废弃。
19. Rewrite winner 只获得参与全局 positive top-\(K_t\) 的资格，不保证进入下一轮 active skill。

## 11. 参考方法及取舍

- [Trace2Skill](./Trace2Skill.pdf)：采用其成功/失败非对称 analyst、agentic failure diagnosis、并行 trajectory-level proposal 和 batch consolidation；不照搬完整 skill directory 结构。
- [SkillRL](./SkillRL.pdf)：采用其成功模式与失败 counterfactual 分开蒸馏的思想；不采用 category-specific SkillBank 与递归检索框架。
- [WikiSkill](./wikiskill.pdf)：只吸收其保留 proposal 历史、避免重复失败编辑的思想，以轻量 ledger 代替完整 Wiki 层。
- [RESKILL](./ReSkill.pdf)：认可 failure profile 和 conditional revision 的价值；不采用 within-group skill-version mixing、bandit version allocation 或 assertion orchestration。
- [Skill1](./Skill1.pdf)：不采用可训练 skill distiller 与 selection/utilization/distillation 联合目标，避免把 candidate generation 变成额外的核心优化问题。

最终选择可以概括为：

> **Treat trajectory outputs as evidence rather than candidates; compile them many-to-one into a compact skill; separate successes and failures; diagnose failures with privileged, environment-grounded replay; expose only ADD/REWRITE candidates; let randomized validation own every acceptance, replacement, and deletion decision.**
