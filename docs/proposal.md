# ShopSimRL 项目 Proposal

## 基于轨迹分析与 Skill 辅助强化学习的个性化购物 Agent 后训练

> 状态：v0.2，研究设计草案  
> 日期：2026-08-23  
> 研究范围：ShopSimulator Single-Turn & Personalization  
> 训练模型：Qwen3.5-4B  
> Teacher / LLM-as-a-Judge：Qwen3.5-397B-A17B

## 1. 项目摘要

本项目研究如何通过后训练提升小参数语言模型在个性化购物环境中的 agentic 能力。实验以 ShopSimulator 的 Single-Turn & Personalization 场景为环境，以 Qwen3.5-4B 为训练模型，先建立 SFT + GRPO 基线，再探索由轨迹分析驱动的 skill-assisted ICRL 方法。

项目不预先规定 skill 的类别、表达形式、注入位置、评估指标或 curriculum control 规则。我们将先对 Qwen3.5-4B 及其 SFT checkpoint 在训练任务上的成功与失败轨迹进行系统分析，再根据观察到的能力缺口和错误模式决定：

- 什么信息值得被抽象为 skill；
- skill 应如何构造、组织和匹配任务；
- skill context 应如何呈现给 agent；
- 哪些指标能够衡量 skill 的即时帮助、长期内化和副作用；
- curriculum control 应如何保留、调整或撤除 skill。

当前计划只进行三次核心训练：

1. Qwen3.5-4B → SFT；
2. 同一个 SFT checkpoint → GRPO；
3. 同一个 SFT checkpoint → skill-assisted ICRL。

最终比较将在不提供 skill 的固定测试集上进行。GRPO 与 ICRL 的公平对照条件、具体训练超参数及计算预算将在环境闭环、数据质量和短程实验完成后冻结，而不在现阶段提前写死。

## 2. 研究背景、目标与边界

### 2.1 研究背景

ShopSimulator 包含商品搜索、属性匹配、规格选择、价格约束和个性化信息利用等购物任务。其论文结果表明，SFT 与 agentic RL 的结合可以明显提升购物 Agent 表现，但小模型在搜索、比较、约束核验、个性化偏好使用以及长轨迹决策上仍可能存在系统性缺陷。

SKILL0 提出一种 ICRL 思路：训练期间向 Agent 提供 skill context，并在训练过程中根据 skill 的作用动态调整其使用，最终在无 skill 的条件下评估模型。本项目借鉴这一方向，但不直接照搬 SKILL0 的 skill 表达、context rendering、helpfulness 定义或 curriculum schedule，而是让 ShopSimulator 轨迹证据决定具体方法。

### 2.2 项目目标

- 在 Qwen3.5-4B 上建立可信的 SFT + GRPO 基线；
- 构建一套可复现的 ShopSimulator persona 数据清洗、重划分和轨迹生产流程；
- 利用成功与失败轨迹识别小模型的关键能力缺口；
- 基于轨迹证据设计适合购物 Agent 的 skill、context 与 curriculum 机制；
- 比较 SFT、GRPO 与 ICRL 在固定、无 skill 测试条件下的表现；
- 分析 ICRL 是否真正改善决策能力，而非只改变输出格式或轨迹长度。

### 2.3 核心研究问题

1. SFT 和 GRPO 分别能够改善哪些购物能力，又会留下哪些稳定失败模式？
2. 成功与失败轨迹中是否存在可压缩、可迁移、可被小模型利用的策略性知识？
3. 如何把这些知识构造成不会泄漏具体答案、又能实际帮助探索的 skill context？
4. skill 的即时帮助是否能够在训练中转化为无 skill 条件下的能力提升？
5. 与相同 SFT checkpoint 出发的标准 GRPO 相比，ICRL 是否能获得更好的最终效果或更高的训练效率？

### 2.4 研究范围

纳入范围：

- ShopSimulator Single-Turn & Personalization；
- 纯文本 observation、action、user profile 与可选 skill context；
- Qwen3.5-4B 的 SFT、GRPO 与 skill-assisted ICRL；
- 环境规则奖励、轨迹级分析和补充性的 LLM-as-a-Judge；
- 最终无 skill 推理和评估。

当前不纳入：

- 多轮 shopper-agent 对话；
- 图片、视频或 SKILL0 的视觉 context rendering；
- 推理期长期记忆系统；
- 多种 RL 算法或模型规模的大规模横向比较。

## 3. 总体研究路线

本项目按“数据 → 轨迹 → 诊断 → 方法 → 对照实验”的顺序推进：

```text
全部 personalization 数据
          │
          ▼
数据清洗、有效性验证、去重与泄漏控制
          │
          ▼
高质量 persona 数据集
          │
          ├── train：N_clean - 800
          ├── val：400
          └── test：400
          │
          ▼
Qwen3.5-397B-A17B 在 train 上随机采样轨迹
          │
          ├── 高质量成功轨迹 → SFT 候选数据
          └── 成功 + 失败轨迹 → 轨迹分析语料
          │
          ▼
SFT checkpoint 与标准 GRPO baseline
          │
          ▼
基于轨迹证据设计 skill / context / metrics / curriculum
          │
          ▼
ICRL 训练与无 skill 固定测试
```

这一顺序是项目的核心约束：在轨迹分析完成前，不冻结 skill taxonomy、SkillBank 格式、context 模板、helpfulness 指标或 curriculum schedule。

## 4. 数据方案

### 4.1 数据范围与重新划分原则

本项目不沿用 ShopSimulator 官方 personalization train/test 划分，而是合并当前已提取的全部 persona 数据，在完成清洗和筛选后重新划分。

当前本地数据审计得到：

| 来源 | 当前可见数量 |
|---|---:|
| 官方 personalization train 文件中的有效 persona 记录 | 3,323 |
| 官方 personalization test 文件中的 persona 记录 | 1,343 |
| 合计 | 4,666 |

论文报告的 train 数量为 3,383，与本地实际提取结果相差 60 条。项目以后续可追溯的数据文件、清洗日志和 manifest 为准，并在报告中说明与论文统计的差异。

数据划分必须发生在清洗之后。设清洗后的高质量数据量为 `N_clean`：

| 子集 | 数量 | 用途 |
|---|---:|---|
| `train` | `N_clean - 800` | Teacher 采样、SFT 数据筛选、轨迹分析、skill 构造、GRPO/ICRL 训练 |
| `val` | 400 | 开发评估、方法选择及可能的 curriculum control |
| `test` | 400 | 配置冻结后的最终评估 |

划分使用固定随机种子。随机选择之前，必须先处理完全重复、近重复以及可能造成跨集合泄漏的关联记录；必要时采用 group-aware random split，使同一重复组不会被拆到不同集合。划分结果保存 task ID、来源、随机种子、分组字段和文件哈希。

`test` 在划分后立即锁定，不参与 Teacher 采样、SFT 筛选、轨迹分析、skill 构造、prompt 修改、Judge rubric 调整或 checkpoint 选择。`val` 可以进入开发和 curriculum 决策，因此不作为最终无偏结果来源。

### 4.2 数据清洗与高质量准入

清洗的目标不是单纯删除格式异常，而是得到能够在 ShopSimulator 中稳定执行、奖励可信且个性化信息可解释的任务集合。准入检查至少覆盖：

- schema、必填字段、类型和编码一致性；
- task、profile、商品及目标属性之间的关联完整性；
- 目标商品、目标规格和选项能否被环境检索与选择；
- target attributes、options、价格和 reward 字段是否相互一致；
- persona 信息是否为空、矛盾、损坏或无法用于当前任务；
- 环境 reset、action parser、termination 与 reward 是否可以稳定复现；
- 完全重复、近重复和模板化样本；
- 可能导致 train/val/test 泄漏的 task、商品、用户或文本关联。

每条被排除或修复的记录都应保留原因码，不静默删除。清洗完成后输出：

- `clean_persona.jsonl`；
- `rejected_persona.jsonl`；
- 清洗规则版本；
- 清洗前后统计与领域分布；
- `train/val/test` split manifest；
- 可复现的数据审计报告。

### 4.3 Teacher 与 Judge

项目使用 **Qwen3.5-397B-A17B** 作为：

1. 训练集轨迹采样的 Teacher；
2. SFT 候选轨迹的语义质量辅助评审模型；
3. 后续轨迹分析、错误归纳和 skill 构造的辅助模型；
4. 环境规则指标无法覆盖时的 LLM-as-a-Judge。

选择同属 Qwen3.5 系列的大模型，是因为它与 Qwen3.5-4B 可能具有更接近的指令理解、工具调用和输出分布，从而降低 Teacher 轨迹与 Student 行为接口之间的不匹配。这是一个需要验证的工作假设，而不是默认成立的事实。

同系列 Teacher/Judge 也可能带来风格偏好和自我偏置。因此：

- 环境可验证的 action、reward 和成功条件优先于 Judge 判断；
- Judge 主要用于规则指标难以覆盖的语义质量、个性化合理性和错误类型；
- Judge prompt、rubric、模型版本、原始输出和解析结果必须留档；
- 在正式使用 Judge 指标前，需要用人工抽检估计一致性和偏差；
- Judge 不得接触锁定 test 的结果来反复修改方法。

### 4.4 轨迹采样与 SFT 数据筛选

Teacher 只在 `train` 中随机采样任务并生成轨迹。采样应保留完整的成功和失败结果，包括模型原始输出、环境 observation/action、终止原因、各 reward 分量、token 用量和采样配置。

轨迹产生后分成两个用途不同的数据视图：

- **SFT 候选池：** 从环境验证成功且质量检查通过的轨迹中筛选；
- **轨迹分析池：** 同时保留成功和失败轨迹，用于能力诊断与方法设计。

SFT 数据允许：

- SFT 与后续 RL 在 `train` prompt 层面重合；
- 根据覆盖不足的领域或难度增加采样。

但需要控制单个 prompt 的轨迹权重、删除高度重复轨迹，并报告唯一 prompt 覆盖率、每 prompt 轨迹数、领域分布、难度分布和轨迹长度分布。

ShopSimulator 论文中的约 6,000 条成功轨迹作为规模参考，而不是现阶段硬性目标。最终 SFT 数量由 Teacher 成功率、唯一任务覆盖、轨迹多样性、人工抽检和小规模 SFT 学习曲线共同决定，避免为了凑数而过度重复简单任务。

### 4.5 轨迹分析

轨迹分析是后续方法设计的依据，而不是 SFT 数据生产的附属步骤。分析对象至少包含：

- Teacher 的成功与失败轨迹；
- Qwen3.5-4B base model 的轨迹；
- SFT checkpoint 的成功、部分成功与失败轨迹；
- 必要时的短程 GRPO 诊断轨迹。

分析重点包括：

- 错误发生在感知、搜索、比较、约束核验、个性化推理、动作执行还是终止阶段；
- 失败是缺少知识、缺少策略、上下文利用不足、格式错误还是探索效率问题；
- 哪些成功行为能够跨商品、领域和 persona 迁移；
- 哪些失败模式可以由文本 context 改善，哪些更可能需要奖励、环境或训练分布调整；
- skill 是否可能诱发过度依赖、错误迁移、上下文干扰或答案泄漏；
- 现有环境 reward 对真实购物质量覆盖不足的部分。

轨迹分析应结合环境统计、规则标签、Qwen3.5-397B-A17B 辅助归纳和人工抽检，形成带有轨迹证据的错误分类体系。该体系将决定后续 skill 和评估设计，但不要求预先固定类别数或层级结构。

### 4.6 数据与轨迹阶段的退出条件

进入正式 SFT 或 RL 前，至少需要满足：

- 高质量数据集和三份 split 已冻结并可复现；
- test 已锁定，训练代码无法默认读取；
- 随机任务可以稳定 reset、执行和计算 reward；
- Teacher 轨迹能够被环境重放，成功/失败判断可追溯；
- SFT 候选池的覆盖、重复率和质量达到可接受水平；
- 成功与失败轨迹已经形成初步、证据化的错误分类；
- 尚未由证据支持的 skill/context/curriculum 设计保持开放。

## 5. 后续方法设计原则与阶段性计划

### 5.1 基线与改进方法

项目保留以下总体实验结构：

- **SFT：** 使用筛选后的成功轨迹建立冷启动 Agent；
- **GRPO baseline：** 从 SFT checkpoint 出发进行标准 agentic RL；
- **ICRL improved：** 从同一个 SFT checkpoint 出发，在训练期引入由轨迹分析产生的 skill context，并最终在无 skill 条件下评估。

GRPO 和 ICRL 的正式公平性条件将在轨迹长度、reward 分布、显存和吞吐 benchmark 后冻结。核心原则是两者共享训练数据、初始化、环境、奖励和主要优化预算；ICRL 引入的额外 skill 生成与 curriculum evaluation 成本需要单独记录。

### 5.2 轨迹分析后再决定的事项

以下内容当前只定义研究问题，不预设答案：

**Skill 构造**

- skill 从成功经验、失败纠正、成功/失败对比还是其他证据中产生；
- skill 是原子规则、步骤化策略、错误提醒、案例摘要还是混合结构；
- SkillBank 的粒度、数量、层级、去重与版本机制；
- 如何防止包含商品 ID、目标答案或任务专属信息。

**Context 机制**

- 使用全局 skill、任务匹配 skill、动态选择 skill 或组合 skill；
- skill 放置位置、长度、格式以及与 system/user/observation 的关系；
- 是否需要任务到 skill 的显式映射、检索或路由；
- 如何测量 context 干扰和模型对 skill 的依赖。

**评估指标**

- 哪些环境 reward 分量作为主指标和诊断指标；
- 如何评价个性化合理性、约束满足、搜索质量、效率和错误严重度；
- LLM-as-a-Judge 适用于哪些维度，如何校准其可靠性；
- 如何区分即时 with-skill 增益与最终 skill-free 能力提升。

**Curriculum control**

- skill 的启用、保留、排序、调整和撤除依据；
- curriculum 使用 val 全集还是固定子视图；
- 控制频率、skill budget、稳定性判断与停止条件；
- 如何控制额外 validation rollout 带来的计算不公平；
- 是否需要完全复用 SKILL0，或根据 ShopSimulator 的连续 reward 和组合任务进行修改。

这些决定必须引用轨迹统计、代表性案例和小规模验证结果，并在正式 ICRL 训练前冻结。

### 5.3 阶段性计划

1. 完成全部 persona 数据的清洗、重放验证和质量筛选；
2. 以固定随机种子生成 `train / val / test`，锁定 400 条 test；
3. 接入 Qwen3.5-397B-A17B，进行小规模 Teacher/Judge 可靠性验证；
4. 在 train 上随机采样成功与失败轨迹，构建原始轨迹池；
5. 筛选 SFT 数据并通过小规模实验确定合理数据规模；
6. 完成 SFT，采样 base/SFT 模型诊断轨迹；
7. 建立有证据支撑的错误分类和能力分析；
8. 据此设计并小规模验证 skill、context、指标与 curriculum；
9. 冻结正式 GRPO/ICRL 对照条件；
10. 依次完成 GRPO、ICRL 和锁定测试集评估。

当前 proposal 到此为止。具体训练超参数、skill schema、context 模板、Judge rubric、评估指标、curriculum schedule、训练基础设施、GPU 预算和统计检验将在对应前置证据完成后，通过版本化实验设计文档另行冻结。

## 参考资料

1. *ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants*. [本地论文](./shopsimulator.pdf)。
2. *SKILL0: Learning Agentic Skills from Zero Data*. [本地论文](./Skill0.pdf)。
