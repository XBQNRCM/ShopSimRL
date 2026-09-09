# IntraSkill：基于 Residual Competence Frontier 的 Chunk-wise Skill-Model Co-evolution

> - 状态：研究方法草案
> - 日期：2026-08-31
> - 当前场景：ShopSimulator Single-Turn & Personalization
> - 核心产物：共同演化的一对 `(final model checkpoint, final skill)`
> - 本文边界：聚焦问题定义、方法机制与文献定位；不展开工程实现、训练配置、实验方案或消融设计。

> 实验实现对照：本次结果见 [experiment_report.md](experiment_report.md)。实际 q 固定为 0.20，failure frontier 使用 all-wrong 判定；本文更一般的前沿或自适应控制讨论属于方法设计空间，不是已完成实验的设置。

## 0. 核心想法

本项目研究一种 **skill 与模型共同演化** 的 agentic RL 范式。它不把 ShopSimulator 中的不同商品类目视为不同技能，而把整个“个性化购物”视为一个任务族，并为这个任务族维护一个统一的 skill。真正需要动态选择、评估和更新的，不是多个彼此近似的 skill 文件，而是同一个 skill 内部承担不同程序性作用的 **skill chunks**。

核心视角是 **residual competence frontier（剩余能力边界）**：模型参数存储已经内化的程序性能力，active skill 则存储对当前 checkpoint 仍有增量帮助、尚未被充分参数化吸收的剩余策略。Skill 不是一份固定教材，而是当前模型能力边界的外部、可读表示；模型每吸收一部分已有知识，skill 的对应边界就收缩，而新的失败模式又会推动它向尚未覆盖的区域扩张。

方法包含两个相互闭环的过程：

- **模型演化：** 训练使用双层 curriculum。全局的 skill-free group probability 控制整体撤除压力；在其余 assisted groups 内，chunk contribution 决定有限辅助预算优先分配给哪些 chunks。每个 GRPO group 始终固定一份上下文，使模型既能借助有效策略探索到正奖励，又持续接受部分或全部指导缺失时的训练。
- **Skill 演化：** 从低回报、低组内差异的 failure frontier 出发，先用完整 current skill 判断失败属于“模型尚未内化已有知识”还是“skill 本身存在 coverage gap”；只对后者生成候选经验。随后在固定 skill-validation split 上通过外生随机 mask 估计每个现有或候选 chunk 对当前模型的边际贡献，据此接纳、重写或淘汰 chunks。

这一过程形成一个自调节课程：**当前贡献越大的 chunk，说明它既有助于提升回报，又尚未被当前模型充分内化，因此在 assisted curriculum 中应更频繁地充当探索脚手架；独立的 skill-free curriculum 则保证模型不会因高贡献而永久依赖它。** 随着模型逐渐吸收该策略，它的增量贡献下降、辅助曝光减少，最终退出 active skill；与此同时，完整 skill 也无法解决的新失败前沿会推动 skill 增长或改写。

## 1. skill 在项目中的定义

### 1.1 四个不同层级

| 层级 | 在本项目中的含义 | ShopSimulator 示例 |
|---|---|---|
| 任务族（task family） | 共享目标结构、环境协议、工具集合和核心工作流的一类任务 | 个性化商品搜索、比较、规格选择与购买 |
| 任务实例（task instance） | 任务族中的一次具体请求 | “为该用户购买预算内的某款羽毛球鞋” |
| 表面变体（surface variation） | 不改变核心解法的内容变化 | 鞋、服装、电子产品、家居用品等商品类目 |
| Skill chunk | skill 内部一个可独立干预和归因的程序性知识单元 | 搜索式构造、persona 证据使用、候选核验、变体价格复核 |

这里最重要的区分是：**商品 category 是任务实例的表面变化，不天然构成 skill 边界。** 如果不同 category 使用相同的页面协议、搜索与点击工具、约束核验逻辑和购买终止条件，那么它们属于同一任务族，应共享一个 shopping skill。

相反，只有当两类任务在操作协议或解题工作流上足够可分时，才有理由建立不同 skill。例如，网页购物、具身环境中的物体操作、开放域搜索问答可以分别拥有自己的 skill。这里更合适的说法是“任务族之间具有清晰的操作边界”，而不是要求它们在数学意义上完全正交。

### 1.2 一个 skill，多个内部 chunks

对当前项目，skill 可以表示为一个结构化 Markdown 文档：

\[
S_t = \{c_{t,1}, c_{t,2}, \ldots, c_{t,M_t}\},
\]

其中 \(S_t\) 是训练时刻 \(t\) 的 shopping skill，\(c_{t,j}\) 是第 \(j\) 个 chunk。一个可能的结构是：

```text
Shopping Skill
├── Query formulation
│   └── 区分硬约束与软偏好；先用商品类型和关键硬约束召回，再迭代补充属性
├── Persona-grounded preference reasoning
│   └── 只使用与当前购买相关的 profile 证据；显式需求优先于推断偏好
├── Candidate inspection and comparison
│   └── 区分标题证据、详情证据与规格选项证据，逐项核验未满足的约束
├── Variant and price verification
│   └── 选定准确规格后重新检查变体价格，不能用商品页默认价格替代
└── Backtracking and purchase termination
    └── 发现硬约束冲突时及时退出；购买前完成最终一致性检查
```

这里的“原子性”是 **干预层面的原子性**，不等于每个 chunk 只能有一句话。一个 chunk 可以包含一个不可拆分的短流程或 checklist，只要这些步骤共同解决同一种失败机制，并且在随机 mask 时应整体出现或整体隐藏。若两个子策略可以独立生效、独立失效或产生不同因果贡献，就应拆成两个 chunks。

每个 chunk 应满足以下语义约束：

- 描述可迁移的程序性知识，而不是某个商品、task ID 或目标答案；
- 具有明确的适用条件、行动原则和常见失败模式；
- 粒度足以被独立遮蔽和评估，又不把一个强耦合流程机械拆散；
- 不包含环境私有 goal、测试信息或无法从公开 observation 获得的事实。

## 2. 方法状态与学习目标

在训练时刻 \(t\)，系统维护以下共同变化的状态：

| 状态 | 记号 | 含义 |
|---|---|---|
| Agent policy | \(\pi_{\theta_t}\) | 当前模型 checkpoint |
| Skill contents | \(S_t\) | 当前保留的 chunks 及其文本 |
| Chunk contribution | \(\alpha_{t,j}\) | chunk \(j\) 对当前模型的 paired-delta 加性贡献估计 |
| Skill-free probability | \(q_t\) | 一个训练 group 完全不提供 skill 的概率，即全局内化压力 |
| Assisted inclusion | \(\rho_{t,j}\) | 在 assisted group 内提供 chunk \(j\) 的概率 |

方法并不把 \(\alpha_{t,j}\) 视为 chunk 的永久质量分数。它是一个 **checkpoint-relative、distribution-relative** 的 residual utility：同一条建议对弱模型可能很重要，对已经掌握该策略的模型可能没有增量价值；模型进一步演化后，它也可能因策略冲突而变成负贡献。\(q_t\) 与 \(\rho_{t,j}\) 刻意分工：前者决定“总体上撤掉多少辅助”，后者决定“剩余辅助预算优先给谁”。

因此，模型与 skill 的关系不是“先生成 skill，再用它训练模型”的单向流水线，而是：

\[
\pi_{\theta_t}
\rightarrow \text{暴露当前失败前沿}
\rightarrow S_{t+1}
\rightarrow \text{改变探索分布}
\rightarrow \pi_{\theta_{t+1}}.
\]

最终目标也不是单独得到一个更强模型或一个更长的 skill 文档，而是得到彼此适配的一对产物：

\[
(\pi_{\theta_T}, S_T).
\]

## 3. 总体框架

```text
训练集上采样约 400 条 Qwen3.5-4B 轨迹
                    │
                    ▼
        Trace2Skill 归纳统一的初始 shopping skill
                    │
                    ▼
        规范化为可独立 mask 的 skill chunks
                    │
                    ▼
      同 checkpoint bare val + 随机 masked val，配对估计初始贡献 α₀
                    │
                    ▼
      初始化全局 skill-free 概率 q₀ 与 assisted 分配 ρ₀
                    │
                    ▼
    先按 qₜ 选择 skill-free / assisted group
                    │
                    └── assisted 时再按 ρₜ 采样一次 chunk mask
                                      │
                                      ▼
                    同一 GRPO group 共享同一上下文
                                      │
                                      ▼
                         model update：θₜ → θₜ₊₁
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     有 reward variation                                failure frontier
       交给 GRPO 学习                               （低均值、低组内差异）
                                                              │
                                                              ▼
                                               用完整 current skill 复核
                                                    ┌─────────┴─────────┐
                                                    ▼                   ▼
                                                  成功                  仍失败
                                             model deficit          skill deficit
                                             调整 q / ρ，不新增      Trace2Skill candidates
                                                    └─────────┬─────────┘
                                                              ▼
                                      复用当前 checkpoint bare val，联合随机化配对评估
                                                              │
                                                              ▼
                                       更新 αₜ₊₁、Sₜ₊₁、qₜ₊₁ 与 ρₜ₊₁
                                                              │
                                                              └──── 下一轮
```

这个框架刻意把两种统计任务分开：

1. **GRPO group 负责在相同条件下比较不同轨迹，更新模型。**
2. **专用 skill-validation split 上的随机化 intervention 负责比较不同 chunk 条件，更新 skill。**

模型优化和 skill intervention 归因因此承担不同统计角色：前者比较同一条件下的轨迹质量，后者比较条件本身的效果。

## 4. 阶段 A：Skill 冷启动

### 4.1 轨迹来源

从冻结训练集中随机抽取约 400 个任务实例，使用初始 Qwen3.5-4B policy 生成完整轨迹。成功和失败轨迹均保留，因为两者提供不同信息：

- 成功轨迹揭示当前模型已经能够执行、但值得抽象和稳定复用的工作流；
- 失败轨迹揭示搜索、persona 使用、约束核验、选项选择和终止决策中的系统性缺口。

这一步只使用 train split。专用且固定的 skill-validation split 留给后续 chunk contribution 估计；model-dev 与 test 都不参与 skill 构造、筛选或课程控制。

### 4.2 从轨迹归纳统一 skill

使用 Trace2Skill 风格的 many-to-one 归纳，将大量局部轨迹经验合并成一个统一的 shopping skill。这里采用 Trace2Skill，不是为了建立按任务实例检索的 memory bank，而是利用它的三个关键能力：

1. 分别分析成功与失败轨迹；
2. 把单条轨迹中的局部 lesson 写成可操作的 patch；
3. 跨轨迹合并、去重和解决冲突，得到可迁移的程序性文档。

初始输出随后被规范化为 chunks。原始文档中的重复提醒应合并；同时包含多个可分离机制的长段落应拆开；必须共同执行的步骤则保留为一个复合原子 chunk。

### 4.3 初始贡献校准

初始 skill 不直接被视为可信真值。它和后续候选一样，要经过固定 skill-validation split 上的随机化 masked-skill 评估。该评估给出初始 \(\alpha_{0,j}\)，并据此建立 assisted curriculum 内的第一版 chunk allocation \(\rho_{0,j}\)；全局 skill-free probability \(q_0\) 则独立控制初始撤除压力。因此，Trace2Skill 负责 **提出并组织知识**，随机化验证负责 **决定哪些知识构成当前模型的 residual frontier**。

## 5. 阶段 B：模型演化——Group-consistent Masked-Skill GRPO

### 5.1 双层采样：先决定是否撤除，再分配辅助内容

对一个训练任务 \(x_i\)，先在 group 粒度采样是否完全撤掉 skill：

\[
z_i\sim\mathrm{Bernoulli}(1-q_t),
\]

其中 \(z_i=0\) 表示 skill-free group，\(z_i=1\) 表示 assisted group。只有在 assisted group 中，才进一步为每个 chunk 采样二元 mask：

\[
m_{i,j}\mid z_i=1
\sim
\mathrm{Bernoulli}(\rho_{t,j}),
\qquad
S_t(m_i)=\{c_{t,j}\mid m_{i,j}=1\}.
\]

因此，chunk \(j\) 在任意训练 group 中的边际出现概率是：

\[
P(c_{t,j}\text{ visible})=(1-q_t)\rho_{t,j}.
\]

所有可见 chunks 按稳定顺序拼成一份 masked skill，并与任务说明、persona 和环境协议一起构成该任务的 system prompt。**Skill-free / assisted 决策和 chunk mask 都在 prompt/group 粒度采样一次，而不是为 group 中每条 rollout 单独采样。**

若 GRPO 对同一任务生成 \(G\) 条轨迹，则：

\[
\tau_{i,1},\ldots,\tau_{i,G}
\sim
\pi_{\theta_t}(\cdot\mid x_i,S_t(m_i)).
\]

这 \(G\) 条轨迹看到完全相同的任务、persona、初始 observation 和 masked skill，只因 policy sampling 的随机性而产生不同答案。不同任务或不同 GRPO groups 可以采到不同 masks。

### 5.2 为什么必须在 group 内固定 mask

Canonical GRPO 的相对 advantage 回答的是：**在相同 prompt 条件下，这条轨迹比同组其他随机轨迹好多少？** 如果同一个 group 内一部分 rollout 带 skill、另一部分不带 skill，可以把 reward 概念性地分解为：

\[
R_{i,g}=\mu(x_i,m_{i,g})+\varepsilon_{i,g},
\]

其中 \(\mu(x_i,m_{i,g})\) 是不同 skill treatment 对该任务带来的条件均值，\(\varepsilon_{i,g}\) 是同一条件下由 trajectory sampling 产生的质量变化。混合 context 的 group centering 同时包含：

\[
\underbrace{\mu(x_i,m_{i,g})-\bar\mu_i}_{\text{between-context treatment}}
+
\underbrace{\varepsilon_{i,g}-\bar\varepsilon_i}_{\text{within-context trajectory quality}}.
\]

这并不意味着 heterogeneous-context grouping 对任意目标都必然产生统计偏差：如果优化目标本身就是随机 context mixture，它仍可能对应一种合理的 mixture-level update。但它改变了 vanilla GRPO 的组内比较语义，并把 skill intervention effect 与 response-level credit 耦合在同一个 normalized advantage 中。

本项目把 mask 固定在 group 内，使 \(\mu(x_i,m_i)\) 对同组所有 rollout 相同并在 centering 中抵消，从而恢复 within-context trajectory comparison；chunk treatment effect 则移到独立的 validation gate 中估计。这里主张的是 **credit semantics 与 intervention identification 的分离**，而不是笼统宣称所有混合-context 方法都有偏。

### 5.3 为什么训练时需要 mask

随机 mask 不只是为了“逐渐删掉 skill”，它同时承担四个作用。

**第一，防止上下文依赖。** 如果训练期间始终提供完整 skill，模型只需学会遵循外部文本，不必把策略写入参数；部署时一旦 skill 缺失，就会出现明显的 context shift。

**第二，形成有指导的探索分布。** 在 sparse-reward agentic RL 中，skill 的直接价值是提高产生有效轨迹的概率。不同 chunk 子集对应不同策略先验，使模型能够在多个受指导的探索区域中采样，而不是永远沿着同一份完整脚本行动。

**第三，让“带辅助成功”逐渐过渡为“无辅助成功”。** 当模型先在包含关键 chunk 的条件下获得成功轨迹并接受正向更新后，相同策略可能在后续较稀疏的 mask 下被复现；一旦可以脱离该 chunk 完成任务，就意味着发生了行为层面的内化。

**第四，把 exploration assistance 与 internalization pressure 解耦。** \(\rho_{t,j}\) 决定 assisted groups 中哪些 chunks 更值得使用，\(q_t\) 则直接决定多少 groups 必须完全脱离 skill。训练分布因此始终保留完整 skill、部分 skill 和空 skill 的支持，而不会因为某个 chunk 贡献很高就永久取消无辅助练习。

## 6. 阶段 C：Failure Frontier 与 Model/Skill Deficit 分诊

### 6.1 从 all-zero 推广到低均值、低差异 frontier

对二值成功信号，或以 \(R=0\) 表示完全失败的稀疏奖励，可以先把一个 group 的状态概括为：

| Group pattern | 对模型学习的含义 | 对 skill 更新的处理 |
|---|---|---|
| 全部成功 | 当前条件下已掌握；相对 advantage 退化，但没有明显的新探索缺口 | 不因该 group 生成新 chunk |
| 成功与失败并存 | 存在可区分的轨迹质量，GRPO 可以直接产生有效学习信号 | 优先让 policy optimization 自己学习 |
| 全部为零 | 当前 mask 和当前 policy 下没有发现正奖励轨迹，group-relative advantage 无法指出改进方向 | 进入 failure-frontier 分诊 |

因此，all-zero group 不是一般意义上的“所有失败数据”，而是一个更具体的信号：**当前 policy 在该局部任务区域缺少能够启动学习的探索支点。** 但它只能说明当前 sampled context 下无法学习，尚不能判断问题在模型还是在 skill。

ShopSimulator 的 reward 还包含连续的严格奖励与细分分量。更一般地，failure frontier 应同时关注 group reward mean 与 group dispersion：低均值表示当前策略整体较差，低方差表示 group-relative learning signal 很弱。all-zero 只是“低均值、零方差”的最清晰特例；全组停留在相同低部分奖励时，也应进入同一分诊过程。

### 6.2 用完整 current skill 区分 model deficit 与 skill deficit

对进入 failure frontier 的任务，先在冻结的当前 checkpoint 下提供 **完整 current skill** 做复核。该 intervention 把失败分成三类：

| Sampled context 失败后，完整 skill 的结果 | 诊断 | 后续动作 |
|---|---|---|
| 完整 skill 能产生明显更高回报或成功轨迹 | **Model deficit / internalization deficit**：skill 已覆盖该机制，但模型在内容缺失时还不会复现 | 不生成同义 chunk；保留相关 chunk 的高 residual contribution，同时通过 \(q_t\) 继续施加无 skill 压力 |
| 完整 skill 仍处于 failure frontier | **Skill deficit / coverage gap**：现有外部程序也无法把模型带入可学习区域 | 进入 Trace2Skill candidate generation |
| 失败来自不可观测信息、奖励、工具或数据异常 | **Environment/data deficit** | 退出 skill evolution，交给环境或数据审计 |

这一步使“skill 应该增长”与“模型应该内化”不再由同一个失败标签代理。尤其是，当某个 all-zero group 只是因为高贡献旧 chunk 恰好被 mask 掉时，完整 skill 复核会把它归入 model deficit，避免不断生成重复 chunks。

### 6.3 对 skill deficit 做失败驱动的 Trace2Skill 更新

只有被确认属于 skill deficit 的 groups，才使用完整 current skill、失败轨迹和环境可验证反馈进行 Trace2Skill 风格的根因分析，并形成结构化 edit proposal：

- **ADD：** 当前 skill 没有覆盖某种稳定失败机制，新增一个 chunk；
- **REWRITE：** 原 chunk 的原则正确但条件、步骤或措辞不适合当前 policy，提出替代版本；
- **DELETE：** 发现现有 chunk 明显误导、冲突或鼓励无效行为；
- **MERGE / SPLIT：** 多个 chunks 重复，或一个 chunk 实际包含可独立干预的多个机制。

这些 edit proposals 只是 candidates，不会因来自失败轨迹就自动进入 active skill。它们必须与现有 chunks 一起经过同一套验证门控。

## 7. 阶段 D：随机化 Validation Gate 与 chunk contribution

### 7.1 联合评估当前 chunks 与 candidates

设当前 active chunks 与本轮候选合并后的集合为：

\[
\widetilde S_t = S_t \cup \Delta S_t.
\]

在专用且固定的 skill-validation split 上冻结当前模型 checkpoint。先取得该 checkpoint 完全不带 skill 的逐任务 bare validation reward \(B_i\)，复用这批对外报告 skill-free validation score 的 traces；然后每个任务另运行一条 masked rollout，并对 \(\widetilde S_t\) 中普通 chunk 施加外生随机 mask：

\[
C_{i,j}\sim\mathrm{Bernoulli}(0.5),
\]

其中 \(i\) 表示验证任务，\(j\) 表示 chunk。Mask 由评估器按固定 seed 外生生成；每个任务得到一个联合 mask、一条 masked rollout，以及一条同任务 bare 对照。逐任务相减用于控制任务难度；基线采样噪声仍保留在残差中。

当前 chunks 与 candidates 必须在同一轮、同一 checkpoint、同一验证分布和同一随机化机制下比较。否则，新候选的分数与旧 chunk 的历史分数不在同一个能力基线上，无法直接用于接纳或淘汰。

对于不同 edit 类型：新增内容直接作为新 intervention factor；删除建议通过原 chunk 的低或负贡献体现；替换建议应把 old/new 视为互斥版本进行比较，避免冲突版本在同一 prompt 中同时出现。

### 7.2 逐任务 paired-delta 贡献估计

Cold-start 与训练阶段采用完全相同的估计式：

\[
R_i-B_i=C_i^\top\beta_t+\varepsilon_i,
\qquad
\hat\beta_t=\arg\min_{\beta_t}\sum_i(R_i-B_i-C_i^\top\beta_t)^2.
\]

\(B_i\) 是当前冻结 checkpoint 在同一个 validation task 上完全不带 skill 的实测 reward，\(R_i\) 是随机 mask 下的 reward。训练本来就需要报告 bare validation score，因此直接复用产生该分数的逐任务 traces；同 checkpoint 的已有完整 bare run 不需要重复采样。换 checkpoint 后必须使用新 checkpoint 对应的 baseline，不能沿用旧模型的 \(B_i\)。

回归不加截距、不减全局 bare 均值，只拟合 chunk/version main effects。主因变量是 `reward`/`r_strict` 的逐任务差值；每个辅助 `r_*` 分量分别减去自身的 bare 分量。后文用于选择和 curriculum 的 \(\alpha_{t,j}\) 统一记为 \(\hat\beta_{t,j}\)，即当前 checkpoint 的估计贡献分数。

我们关心的因果目标仍可写为当前 mask 分布下的平均边际贡献：

\[
\tau_{t,j}=\mathbb E[R\mid do(C_j=1)]-\mathbb E[R\mid do(C_j=0)].
\]

但这里的无截距 main-effect 系数是相对于 bare 的加性近似；存在 chunk 交互、异质性或模型失配时，不能仅凭外生随机 mask 就断言 \(\hat\beta_{t,j}\) 等于 \(\tau_{t,j}\)。符号不是“有益/有害”的充分证据，排序价值需要独立验证。

Initial Skill Compiler 将必须共同执行的步骤合并为复合原子 chunk，以改善 main-effect 近似。逐任务相减有望削弱任务难度差异，但 \(B_i\) 也是一次带噪声的观测，是否实际降方差取决于两组 reward 的协方差，不能保证。若做 eligible-slice 分析，必须在观察 reward 前按公开任务属性固定 slice，并保留对应 bare/masked pairs。

实现、配对校验、断点恢复及输出口径见 [paired_validation.md](./paired_validation.md)。两阶段都要求完整同任务配对、同模型/采样/prompt/环境协议；标准 gate 不接受用 test 分数或 summary 均值代替逐任务 baseline。

### 7.3 Contribution gate

贡献度对应三类决定：

- \(\alpha_{t,j}<0\)：当前估计为负，按暂定规则拒绝或退休，但不能仅凭符号断言有害；
- \(\alpha_{t,j}\approx 0\)：当前估计增益小，可降低曝光；可能是内化、冗余或估计噪声，不能仅凭系数确定原因；
- \(\alpha_{t,j}>0\)：进入按有符号系数降序的 top-K 候选池；正值本身不等于显著改善。

从课程控制角度看，不必强行区分“低贡献是因为已经内化”还是“低贡献是因为质量差”：二者都意味着当前 policy 不再需要频繁看到它。为了审计 skill 的知识演化，两种原因可以在文本分析层面保留不同标签，但训练控制使用的是可验证的当前效用。

## 8. 阶段 E：双层 Curriculum——辅助分配与整体撤除

### 8.1 Assisted curriculum：贡献决定有限辅助预算给谁

保留下来的 chunk 在下一轮 assisted groups 中的 inclusion probability 由一个单调映射确定：

\[
\rho_{t+1,j}
=
\operatorname{clip}
\left(
g(\max(\alpha_{t,j},0)),
\rho_{\min},\rho_{\max}
\right),
\]

其中 \(g(\cdot)\) 单调递增；如果需要控制总上下文长度，还可以在保持排序的前提下约束期望 chunk budget。这里不预设具体函数，重要的是以下关系：

\[
\alpha_{t,j}\uparrow
\quad\Longrightarrow\quad
\rho_{t+1,j}\uparrow.
\]

它的理由分成两层：

1. **有效性：** 高正贡献表示该 chunk 能明显提高当前 agent 的任务回报，在有限 assisted budget 中更频繁地提供它，可以增加成功探索和高质量训练轨迹的密度。
2. **未内化性：** 贡献是在同一当前 checkpoint 下比较“有该 chunk”和“无该 chunk”的结果。如果模型已经能从参数中稳定复现该策略，额外提供文本不应继续产生明显增益；持续的正增益说明它仍位于 residual competence frontier。

第二层解释依赖一个重要前提：chunk 必须主要编码可学习的程序性策略，而不是模型无法凭参数恢复的实时事实、私有标签或任务答案。否则，高贡献只能说明外部信息有用，不能说明策略尚未内化。

### 8.2 Skill-free curriculum：独立控制整体内化压力

仅有 \(\rho_{t,j}\propto\alpha_{t,j}\) 只能回答“辅助谁”，不能保证模型最终脱离辅助。为此，\(q_t\) 独立控制完全 skill-free groups 的比例。它对应从 context-supported execution 到 parametric competence 的整体转移压力。

可以用两个冻结条件下的性能定义全局 internalization gap：

\[
G_t
=
\mathbb E[R\mid S_t\text{ full}]
-
\mathbb E[R\mid S_t=\varnothing].
\]

随着 skill-free performance 上升且 \(G_t\) 缩小，curriculum 可以把更多训练质量分配给 skill-free groups；同时，\(q_t\) 应始终保留非零撤除压力，避免系统因短期 assisted return 较高而停留在永久依赖状态。这里不预设具体 controller，只要求它同时满足：早期不过快撤掉探索脚手架，后期不会因高 \(\alpha_j\) 而停止无辅助训练。

### 8.3 为什么高贡献 chunk 不会被永久保留

高贡献导致 assisted curriculum 中更高的 \(\rho_{t,j}\)，并不等于让模型永远依赖它，因为：

- \(q_t>0\) 时，模型始终会遇到完整 skill 缺失的训练条件；
- \(\rho_{t,j}<1\) 时，assisted groups 内也会出现该 chunk 缺失的部分指导条件；
- 在 chunk 出现时，它提高成功轨迹概率，为模型提供可学习的正向行为；
- 当参数逐渐吸收该行为后，有无 chunk 的性能差缩小，\(\alpha_{t,j}\) 自然下降；
- \(\rho_{t,j}\) 随贡献下降，模型获得更多脱离该 chunk 的练习，直到它被退休。

因此这不是单向的“重要内容多看”，而是一个负反馈控制环：

```text
chunk 有用但尚未内化
        ↓
assisted groups 中较高 ρ → 更多受指导的成功探索
        ↓
policy 学会相关行为
        ↓
chunk 的增量贡献下降
        ↓
ρ 降低，同时 q 提供持续的全局无辅助练习
        ↓
充分内化后退出 active curriculum
```

与单一概率或按训练 step 统一减少 skill 数量相比，双层课程分别管理 **exploration assistance** 与 **internalization pressure**：不同 chunks 可以以不同速度被吸收，新产生的有效 chunks 也可以在训练后期进入 assisted curriculum，而全局 skill-free 训练不会因此被取消。

## 9. 一轮完整的共同演化

综合起来，第 \(t\) 轮只包含以下概念步骤：

1. 以 \(q_t\) 决定一个 prompt/group 使用空 skill 还是进入 assisted curriculum；若为 assisted group，再按 \(\rho_{t,j}\) 采样 chunk mask；
2. 对该 prompt 的全部 \(G\) 条 rollout 使用同一份上下文，并执行标准 group-relative policy update；
3. 从 rollout 中识别低回报、低组内差异的 failure frontier；
4. 用完整 current skill 对 frontier 样本进行复核，将其分为 model/internalization deficit、skill coverage deficit 与 environment/data issue；
5. 只针对 skill coverage deficit 调用 failure-focused Trace2Skill，形成新增、改写或合并 candidates；
6. 冻结当前 checkpoint，复用用于报告 bare validation score 的逐任务 traces；在相同 skill-validation split 上对 current chunks 与 candidates 联合随机 masking，用 \(R_i-B_i=C_i^\top\beta+\epsilon_i\) 无截距回归估计 \(\alpha_{t+1}\)；
7. 接纳正贡献 candidates，退休有害、冗余或已内化 chunks，得到 \(S_{t+1}\)；
8. 分别更新 assisted curriculum 内的 \(\rho_{t+1,j}\) 与全局 skill-free pressure \(q_{t+1}\)，进入下一轮。

其中 policy 每个训练 step 都可以更新，而 failure triage、candidate generation 和 validation gate 作为较慢的外循环周期性发生。二者时间尺度不同，但都以同一个当前 checkpoint 为参照：policy 在吸收已有 frontier，skill 则在重估并推进 frontier。

## 10. 从单任务族扩展到多任务族

当前项目只训练 ShopSimulator 这一种任务族，因此只需要一个 shopping skill。方法可以自然扩展到多任务训练集：

\[
\mathcal D
=
\bigcup_{k=1}^{K}\mathcal D^{(k)},
\qquad
\mathcal D^{(k)}\leftrightarrow S^{(k)}.
\]

每个任务族 \(\mathcal D^{(k)}\) 对应一个自己的 skill \(S^{(k)}\)，共享同一个 policy，但各自维护：

- 独立的 chunk 集合；
- 独立的 contribution scores、\(q_t^{(k)}\) 与 \(\rho_{t,j}^{(k)}\)；
- 只属于该任务族的 skill-validation slice；
- 先经完整 \(S^{(k)}\) 复核、再由 skill coverage deficit 触发的 candidate evolution。

当训练 batch 同时包含购物、搜索问答和具身操作时，任务首先按已知任务协议路由到对应 skill；随后只在该 skill 内部随机 mask 和演化。这样，随着训练推进，不是“先内化任务 A 的整个 skill，再内化任务 B”，而是多个任务族各自的能力与 skill 同步前进：

```text
shared policy
├── shopping skill：query / persona / comparison / variant chunks 同步演化
├── search skill：query decomposition / source checking / synthesis chunks 同步演化
└── embodied skill：navigation / object state / manipulation chunks 同步演化
```

跨任务的共享知识仍然可以通过模型参数迁移，但不应为了复用几条泛化建议而抹去 skill 的操作边界。

## 11. 最终产物与推理语义

共同演化不能只用“最终还剩多少 skill”衡量。对最终 checkpoint，至少应区分：

\[
R_{\mathrm{free}}
=
\mathbb E_x[R(x;\pi_{\theta_T},\varnothing)],
\qquad
R_{\mathrm{pair}}
=
\mathbb E_x[R(x;\pi_{\theta_T},S_T)],
\]

以及二者之差所刻画的 internalization gap：

\[
G_T = R_{\mathrm{pair}}-R_{\mathrm{free}}.
\]

其中，\(R_{\mathrm{free}}\) 是模型已经写入参数的独立能力，\(R_{\mathrm{pair}}\) 是当前 model-skill pair 的能力上界，\(G_T\) 则近似表示仍位于外部 residual frontier 中的有效能力。理想的共同演化应推动 \(R_{\mathrm{free}}\) 上升，在不牺牲 \(R_{\mathrm{pair}}\) 的前提下缩小同一阶段的 gap；如果 skill 同时发现了新的有效策略，\(R_{\mathrm{pair}}\) 与 gap 也可能暂时重新上升，这代表 frontier 被向外推进，而不一定是训练退化。

最终输出是：

1. **Final model checkpoint \(\pi_{\theta_T}\)：** 已在空 skill 与多种 assisted contexts 上训练，尽可能内化可迁移策略；
2. **Final task-family skill \(S_T\)：** 经过当前模型、专用 skill-validation 分布和随机化贡献门控筛选的结构化购物程序，表达尚有增量价值的 residual frontier。

这两个产物支持两种互补的推理模式：

- **Skill-augmented mode：** 把完整、静态的最终 Markdown skill 注入 system prompt。此时不再随机 mask，目标是获得模型与其共同演化 skill 的最佳组合表现。
- **Skill-free mode：** 不提供 skill，用于衡量参数内化程度、上下文缺失时的鲁棒性，或部署预算严格受限的场景。

因此，“内化 skill”不要求最终一定销毁 skill。本文把 **skill-free 能力与 gap 收缩** 作为检验内化的主要信号，把 **paired performance** 作为共同演化系统的能力上界；最终 skill 仍作为可读、可审计、可迁移和可继续演化的外部程序保留下来。最终系统的主产物是二者的配对，而不是强行在“纯参数模型”和“纯外部 skill”之间二选一。

## 12. 关键假设与概念边界

### 12.1 Contribution 不是永恒的知识价值

\(\alpha_{t,j}\) 只表示 chunk 对当前 checkpoint 和当前验证分布的局部平均贡献。它不能证明该 chunk 对所有模型、所有任务或所有训练阶段都重要。因此 skill、贡献和模型 checkpoint 必须共同版本化。

### 12.2 低贡献有多种原因

低贡献可能表示已内化、内容重复、触发范围太窄、措辞不适配模型，或策略本身错误。课程控制可以统一降低它们的出现率，但 skill 审计仍应记录原因，避免把“模型已经学会”与“知识本身无效”混为一谈。

### 12.3 Chunk 原子性与 main-effect contribution

Validation-ready chunks 是彼此独立的最小原子干预单元。不可分的强耦合步骤在 compilation 阶段合并为一个复合 chunk；可独立生效的程序性机制分别成为不同 chunks。随机化验证统一使用逐任务 paired-delta 无截距 OLS main effects，\(\alpha_j\) 是相对于同 checkpoint bare validation 的加性贡献估计；交互存在时不能直接等同于平均边际因果效应。

### 12.4 Skill evolution 与 policy update 使用不同数据角色

需要至少区分四种数据角色：

- **Train：** policy learning、failure frontier 发现与 candidate proposal；
- **Skill-validation：** 在冻结 checkpoint 下估计 chunk intervention effects，并驱动接纳、退休和课程控制；
- **Model-dev：** checkpoint 与普通超参数选择；
- **Test：** 只用于最终报告。

固定 skill-validation split 有利于跨轮比较，但它被共同演化过程反复查询，本身也会产生 adaptive overfitting。因此它不能同时充当最终 model-dev 或 test；必要时还应保留未参与在线 skill 决策的 audit slice。这里所谓“验证贡献”始终是对当前 checkpoint 和指定 skill-validation 分布成立的局部量。

### 12.5 这里不解决所有 exploration 问题

Skill 可以把部分低回报、低差异 group 推入可学习区域，但若失败来自环境不可观测、奖励错误、工具缺失或数据标签异常，新增文本 chunk 不会解决问题。这些情况应通过完整-skill 复核与环境审计被隔离，而不是无限扩张 skill。

## 13. Motivation：为什么要这样定义问题

### 13.1 ShopSimulator 的困难本身是跨 category 的

ShopSimulator 的分析显示，购物 agent 的主要困难集中在深度搜索、相似商品区分、细粒度属性和选项满足、persona 信息的平衡使用，以及长轨迹中的决策稳定性；论文也发现 SFT 提供工作流先验，而 RL 进一步改善细粒度需求满足。这些能力横跨服装、鞋类、电子产品和家居用品，并不随商品 category 改变其基本性质。

因此，真正值得学习和内化的是一个跨 category 的购物工作流，而不是多个只因商品名不同就分开的近重复技能文件。商品内容提供 task variation，skill 内部的策略组合提供 learning variation。

### 13.2 对 Skill0 抽象层级的修正

Skill0 的重要贡献是把 skills 当作训练期脚手架，并依据 on-policy helpfulness 动态撤除；其理论也明确指出，当无 skill 表现追上有 skill 表现时，helpfulness 应趋近于零。这个 insight 直接启发了本项目的 contribution-driven curriculum。

问题在于，Skill0 将 SkillBank 按 general / task-specific files 组织，并在 WebShop 示例中进一步按 apparel、electronics、footwear、beauty & health、home decor、accessories 等商品类目拆分文件。对高度同构的 WebShop/ShopSimulator 任务，这些文件中的核心原则——构造搜索式、快速退出不匹配候选、选择准确 variant、复核价格、确认后购买——具有明显重复。

在这种场景中，按 category 绑定 skill 会带来两个概念问题：

1. **Skill 边界与真实工作流边界错位。** 同一类购物能力被复制到多个文件，而 category 差异被误认为技能差异。
2. **File-level curriculum 容易与任务子集耦合。** 先保留某个 category skill、后撤掉另一个 category skill，可以被理解为在不同时间重点辅助不同任务子集，而不是让所有购物能力在同一个任务族内连续、同步地内化。

本项目保留 Skill0 的“帮助度随 policy 改变”和“撤除已内化脚手架”思想，但把控制单位从 **skill files 之间** 下沉到 **一个 task-family skill 内部的 chunks 之间**。课程不再回答“现在训练哪个商品类目的 skill”，而回答“当前 shopping agent 仍需要哪些程序性策略”。

### 13.3 从 skill augmentation 走向真正的共同演化

SkillRL、Skill1 和 WikiSkill 都强调从轨迹积累可复用知识并持续更新 skill，但主要目标仍是维护、检索和利用外部 skill bank：

- SkillRL 组织 general 与 task-specific skills，并在 RL 中递归扩展 SkillBank；
- Skill1 统一训练 skill selection、utilization 和 distillation；
- WikiSkill 在 raw traces 与 executable skills 之间增加可持续积累的 wiki，并用 validation gating 回滚 skill edits。

这些工作说明 skill 不应是静态 prompt，但它们较少把“一个任务族内部哪些知识已被模型内化”作为显式随机干预对象。本项目把共同演化定义为双向变化：skill 根据模型的失败前沿增长，模型根据 skill 提供的探索脚手架成长，而每个 chunk 是否仍有必要由当前 checkpoint 的边际贡献决定。

### 13.4 近期共进化工作如何改变创新性边界

“训练模型的同时更新 skill”本身已经不再是空白。近期工作从不同方向推进了这一主题：

- **Co-Evolving Skill Generation and Policy Optimization** 在 policy optimization 过程中持续生成 skills，并在入库前比较 candidate 的边际效用；
- **D2Skill** 维护 task-level 与 step-level 的双粒度动态 skill bank，并利用 baseline/skill-enhanced trajectories 协调 skill 使用与 policy learning；
- **RESKILL** 在 old/new skill versions 之间进行在线比较，以 bandit 机制控制 skill 更新；
- **SPyCE** 将 skill-policy co-evolution 扩展到多模态 agent，并在训练中持续更新 execution/workflow 两级 skill library。

因此，本项目不应把“co-evolution”“细粒度 skill”或“marginal utility”单独包装成创新。更准确的切入点是一个尚未被这些工作直接解决的边界问题：**如何识别当前能力中哪些部分应继续留在外部 skill，哪些部分已经可以转移到模型参数，以及 skill 的下一步增长究竟应修补外部知识还是继续训练模型。**

本项目为这个问题引入三个彼此耦合的结构：task-family 内的 residual competence frontier、分离的 \(q_t/\rho_{t,j}\) 双层课程，以及通过完整 skill 复核实现的 model/skill deficit 分诊。创新性如果成立，应来自这一整套能力边界建模与控制闭环，而不是来自任一孤立组件。

### 13.5 为什么把 GRPO credit 与 skill intervention 分开

EDGE 把同一 GRPO group 划分为 experience-conditioned teacher rollouts 与 experience-free student rollouts，并用二者差异估计 marginal gain；RESKILL 在同一 group 内为不同 rollouts 分配 old/new skill versions；D2Skill 也显式构造 baseline 与 skill-enhanced trajectories。它们的共同优点是复用训练 rollout 做 skill evaluation，但同一 group 中的样本不再共享完全相同的 prompt context。

如第 5.2 节所述，若

\[
R_{i,g}=\mu(x_i,m_{i,g})+\varepsilon_{i,g},
\]

那么混合-context group 的中心化 reward 同时包含 context treatment 造成的 \(\mu\) 差异与同一 context 下 trajectory quality 造成的 \(\varepsilon\) 差异。如果优化目标本来就是这种 context mixture，这样的 estimator 未必在统计意义上“有偏”；但它不再具有 vanilla same-context GRPO 中纯粹的 within-prompt response comparison 语义，skill evaluation 与 policy credit assignment 也被绑定在同一个标量中。

本项目主张的是一种 **separation principle**：

- **Policy update：** 每个 group 固定同一份 sampled context，只让生成轨迹变化，保留清晰的 within-context relative advantage；
- **Skill intervention：** 在冻结 policy 的专用 skill-validation gate 中外生改变 chunks，单独估计 checkpoint-relative treatment effects。

这不是宣称混合-context 方法普遍无效，而是选择更容易解释和审计的 credit semantics。Skill-SD 则从另一方向处理 skill-conditioned 与 plain prompt 的分布差异：让 skill 只条件化 teacher，并通过 importance-weighted self-distillation 向 plain student 转移能力。本项目不引入额外蒸馏分支，而让 RL policy 直接覆盖空 skill 与多种 assisted contexts。

### 13.6 Trace2Skill 在本项目中的准确角色

Trace2Skill 证明了广泛的成功/失败轨迹可以被 many-to-one 地压缩为一个可迁移 skill，并且局部 patches 经过合并后常形成稳定 SoP。它非常适合两个环节：冷启动统一 skill，以及从新失败前沿提出 edit candidates。

Trace2Skill 生成结果先经过 many-to-one compilation，把不可分的局部 patches 合并为复合原子 chunk，再进入联合随机 mask。现有 chunks 与 candidates 由同一个 randomized validation experiment 和 paired-delta 无截距 OLS 统一估计贡献，复用同 checkpoint 的 bare validation traces。

## 14. 潜在创新点

在近期相关工作的背景下，本文最有分量的贡献不是“又一种会更新的 skill bank”，而是把 **模型参数与外部 skill 之间的动态能力边界** 变成可估计、可诊断、可控制的训练状态。下面各点应作为一个整体来主张；其中任一组件单独拿出，都可能与已有工作发生较强重叠。

### 14.1 Residual competence frontier as co-evolution state

把 active skill 重新定义为当前 checkpoint 的 residual competence frontier：它不代表任务的全部解法，而代表仍能为模型提供正增量、尚未被稳定参数化的那部分程序性知识。模型内化会使 frontier 收缩，新的 skill coverage 会使 frontier 外扩，从而为“共同演化了什么”提供统一状态语义。

### 14.2 Task-family-aligned intra-skill evolution

把 skill 边界对齐到共享环境协议与解题工作流的任务族，而不是 benchmark 中的表面 category；在一个 task-family skill 内以 chunks 表达真正的策略 variation。这样，搜索、persona 推理、候选比较与购买核验可以同步演化，但各自以不同速度被内化。

### 14.3 Separation of policy credit and skill attribution

训练时采用 group-consistent context masking，保持 same-context group-relative advantage；训练外在冻结 checkpoint 和专用 skill-validation split 上施加随机 chunk interventions。该设计把 response-level policy credit 与 chunk-level treatment attribution 分开，使二者各自具有清晰的统计语义。

### 14.4 Interventional chunk lifecycle

现有 chunks 与新增 candidates 在同一 checkpoint、同一验证分布、同一联合随机化机制下估计 average marginal contribution。由此统一支持 candidate admission、旧 chunk retirement、冗余识别和组合关系审计，而不是让生成器输出直接成为 active skill。

### 14.5 Dual-control internalization curriculum

用两个不同控制量管理两种不同问题：\(q_t\) 控制完整 skill 被撤除的全局内化压力，\(\rho_{t,j}\) 根据正贡献分配 assisted groups 内的有限上下文。高贡献 chunk 获得更多受指导探索，但持续的 skill-free groups 防止模型把“高贡献”误解为“永久依赖”；贡献下降后 chunk 自动退出 active curriculum。

### 14.6 Counterfactual model/skill deficit triage

把 failure trigger 从 all-zero 推广到低均值、低组内差异的学习前沿，并用完整 current skill 做反事实式复核：若完整 skill 能救回，则是模型内化/探索不足；若仍失败，才是 skill coverage gap；环境与数据异常另行隔离。这避免用同一个失败标签同时驱动“继续训练模型”和“扩张 skill”。

### 14.7 Frontier-aware objective and paired artifacts

以 \(R_{\mathrm{free}}\)、\(R_{\mathrm{pair}}\) 与 internalization gap 联合描述结果：前者反映参数化能力，后者给出 model-skill pair 的能力上界，二者之差刻画仍有增量价值的外部 frontier。最终交付匹配的 model checkpoint 与 executable skill，而不是只汇报一个 skill-conditioned 分数。

## 15. 与相关工作的关系概览

| 工作 | 主要演化对象 | Skill 在训练/推理中的角色 | 与本项目的关键区别 |
|---|---|---|---|
| ShopSimulator | Policy | SFT 提供工作流先验，RL 优化细粒度满足 | 提供环境与问题动机；本项目进一步研究动态 skill 脚手架 |
| SkillRL | Hierarchical SkillBank + policy | 检索 general / task-specific skills，推理时继续使用 | 本项目用一个 task-family skill，并显式追踪 chunk 内化 |
| Skill0 | Skill files 的 active subset + policy | 训练时逐步撤除，推理时归零 | 本项目把课程单位改为 skill 内部 chunks，并保留最终 model-skill pair |
| Trace2Skill | 一个静态 skill directory | 从广泛轨迹归纳并合并 patches | 本项目用其提出 candidates，再以当前 policy 的随机化贡献门控 |
| EDGE | Experience bank + distilled policy | 同 group 比较有/无 experience，推理时移除 | 本项目不在同一 GRPO group 内混合不同 context |
| RESKILL | Old/new skill versions + policy | 同 group 分配版本并用 bandit 接纳/拒绝 | 本项目把版本评估移至 validation，把 group 留给同 prompt policy credit |
| Co-Evolving Skill Generation and Policy Optimization | Candidate skills + policy | 在线生成并在入库前验证边际效用 | 本项目聚焦一个 task-family 内 residual frontier 的持续估计、内化与缺口分诊 |
| D2Skill | Task-/step-level dynamic skill banks + policy | 双粒度检索，并比较 baseline/skill-enhanced trajectories | 本项目不用双库表达任务内 variation，而以统一 skill 内 chunks 与 \(q/\rho\) 双层课程控制能力转移 |
| SPyCE | Multimodal skills + policy | 随 policy 持续更新分层 skill library | 本项目聚焦文本 agentic RL 中参数—外部 skill 边界的随机干预与分诊 |
| Skill-SD | Skill-conditioned teacher + student | Student 始终 plain prompt，通过蒸馏内化 | 本项目不增加蒸馏分支，以 group-consistent masked-skill RL 内化 |
| Skill1 | Selection / utilization / distillation policy + skill library | 外部 skill 持续检索和积累 | 本项目聚焦一个任务族内部的 chunk lifecycle 与 causal contribution |
| WikiSkill | Raw / Wiki / Skill 三层外部知识 | Skill edits 经 validation gating，知识长期保留 | 本项目的持久状态是 chunk 贡献和内化程度，而非额外 wiki 层 |

## 参考文献

1. *ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants*. [本地论文](./shopsimulator.pdf)
2. *SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization*. [本地论文](./Skill0.pdf)
3. *Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills*. [本地论文](./Trace2Skill.pdf)
4. *SKILLRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning*. [本地论文](./SkillRL.pdf)
5. *Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning*. [本地论文](./Skill1.pdf)
6. *EDGE: Experience-Distillation for Guided Exploration in Agentic Reinforcement Learning*. [本地论文](./EDGE.pdf)
7. *RESKILL: Reconciling Skill Creation with Policy Optimization in Agentic RL*. [本地论文](./ReSkill.pdf)
8. *Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents*. [本地论文](./Skill-SD.pdf)
9. *WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution*. [本地论文](./wikiskill.pdf)
10. *Co-Evolving Skill Generation and Policy Optimization*. [arXiv](https://arxiv.org/abs/2606.08755)
11. *Dynamic Dual-Granularity Skill Bank for Agentic RL*. [arXiv](https://arxiv.org/abs/2603.28716)
12. *SPyCE: Skill-Policy Co-evolution for Multimodal Agents*. [arXiv](https://arxiv.org/abs/2607.13854)
