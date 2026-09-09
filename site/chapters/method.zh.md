# 方法：什么在共同演化，为什么

IntraSkill 研究购物策略与外部程序性指导能否共同改进。策略通过 GRPO 学习；另一条循环分析失败、提出可复用指导、测量它对当前 checkpoint 的作用，再修订 active skill。本次发布实验运行了这一完整系统，但没有分离其相对普通 RL 的额外收益。

## 一个任务族，多个程序性技能块

任务族是个性化购物：理解请求和 persona、搜索目录、检查商品、选择规格并购买。鞋与电子产品是工作流中的表面变体，不天然对应不同技能。**Chunk** 是可独立注入或隐藏的知识单元，例如搜索式构造、基于证据的 persona 推理、候选核验、规格选择或购买前价格检查。

Active skill 是这些 chunks 的结构化集合，内容应描述可迁移程序、适用条件和常见失败方式。商品 ID 或私有目标答案不是可复用程序。这一区分很关键：一次成功的诊断重试既可能发现通用策略，也可能只找到了当前任务的捷径。

## 剩余能力边界

模型参数承载已学会的行为。当前 checkpoint 尚未可靠掌握某种程序时，外部 chunk 可能仍有帮助，我们把这种相对当前 checkpoint 的效用称为 *residual competence frontier*（剩余能力边界）。随着模型进步，一条指导的帮助可能下降；在新的失败模式下，也可能重新变得有用。贡献不是永久质量分数。

这是一种研究解释，并非对模型内部知识的直接测量。接近零的回归系数也可能来自噪声、交互、冗余或统计功效不足。技能库收缩本身不能证明内化；本次大小实际上为 10 → 7 → 5 → 8 → 8。

## 快循环：组内一致的技能辅助

每个 GRPO group 包含同一任务的八条轨迹。先以 **q = 0.20** 的概率选择整组无技能；否则，按每个 active chunk 的 rho 采样联合 mask。八条 siblings 共享完整 mask，使组内比较发生在相同指导条件下。若给 siblings 不同指导，group advantage 就会混合上下文差异与策略采样差异。

本次 q 固定。在 assisted groups 中，正贡献被线性映射至 [0.20, 0.90] 的 rho，最大正贡献映射到上界。两个 gate 之间的 curriculum 带哈希冻结。Assisted 分支也可能未采中任何 chunk；runtime 将其与 skill-free 分支单独记录。

```text
任务 + 冻结 curriculum
  └─ 一份 group mask
       ├─ 轨迹 1 ─ reward 1
       ├─ ...
       └─ 轨迹 8 ─ reward 8
             └─ 组内相对 advantage → 策略更新
```

训练保留实际采样的 token ID 和 rollout log probability。Prompt、环境 observation 和工具返回的 loss mask 为零，模型生成 token 承担训练损失。动态工具 schema 可能将一个 episode 拆成多个连续 token 段；奖励归一化先按原 rollout identity 去重，避免长轨迹因分段而在组均值里重复计权。

### 动态过滤及其分母

完整训练脚本每个 rollout step 生成 32 groups、保留 16 groups，即 128 条 episode。Global batch 为 64，因此每步对应两次优化器更新。80 步合计生成 20,480 条、接纳 10,240 条。

含技术故障的组被丢弃并补采。零方差组优先级较低，但凑批时可以保留。因此日志里的零方差**丢弃率**是过滤统计，不等于全部生成组里同奖励组的发生比例。行为分析覆盖生成且可评分的轨迹，不声称重建了 trainer 实际接纳样本的完整分布。

## 慢循环：先诊断失败，再写指导

只有八条轨迹均正常可评分且全部失败的组，才额外用完整 current skill 重试一次。诊断轨迹不进入 GRPO。重试成功提示 internalization deficit：已有指导可以解决 masked group 未解决的情况。重试失败只代表需要进一步分析，不能自动认定技能存在缺口。

Trace2Skill 消费这些失败诊断轨迹。Analyst 可以查看特权诊断信息，但最终可部署指导必须抽象为来自公开 observation 的证据和可复用动作；私有目标事实、标识和 gold answer 不能进入 policy skill。ADD 必须有真实成功的反事实试验，REWRITE 必须指向当前 active chunk，Analyst 也可以返回 NO_PROPOSAL。

轮次结束后，compiler 对 eligible cards 按机制去重，合并为最多六条 ADD/REWRITE candidates。多张 card 可以支持同一个候选。Proposal ledger 记录历史，但不注入策略 prompt。本次候选数为 6、6、6、4；eligible cards 为 61、22、20、11。

## 配对随机验证

每 20 个 rollout step 冻结 checkpoint。通用 evaluator 在 400 个验证任务上运行 bare evaluation；gate 在相同任务、相同 checkpoint 上，对 current chunks 与候选池做随机 mask 评测。Task、sample、模型身份、解码参数、prompt、环境与 action protocol 必须满足配对契约。

对任务 i，Bᵢ 是 bare reward，Rᵢ 是 masked reward，Cᵢ 是实际注入的版本指示向量。拟合：

```text
delta_i = R_i − B_i
beta_hat = argmin_beta Σ_i (delta_i − C_iᵀ beta)²
```

不拟合截距，不中心化 treatment matrix，也不减全局 bare 均值。全零 mask 仍是有效观测：两条随机生成的 bare 等价轨迹也可能 reward 不同。主目标为 strict reward；每个完整奖励分量分别拟合自己的 paired delta。

普通 slot 的 absent/present 等概率；rewrite family 使用互斥的 absent/old/new 状态，竞争版本不会同时出现在 prompt。先在 family 内按系数选唯一 winner，同分保留 old；再做全局 strictly-positive top-K，K ≤ 10。不用非正系数补齐技能库。Selected bank 决定下一轮 curriculum。

### Gate 能说明什么

四轮公开 observation table 的重拟合与保存系数相差不到 2 × 10⁻¹⁶，说明数值产物一致。这不能证明不存在 chunk 交互、服务权重必然正确，或 unseen tasks 上的排序可靠。

Masked validation 均值描述 current-plus-candidate 随机池，**不是**最终选中 full skill 的整体收益。固定 val 被跨轮复用于选择，因此不是未触碰的最终测试集。逐任务相减能否降方差还取决于 noisy bare 与 masked outcomes 的协方差，不能保证。

## 本次回答了什么，仍需什么实验

训练后的 bare policy 在冻结 test 上明显优于初始模型，固定验证任务上的生成行为也发生变化。这与学习到有用程序相容，但本次缺少 plain GRPO、固定 skill、无 mask、无 online gate、不同 q 和多训练种子对照。它们对于归因共同演化的额外收益和验证内化机制仍然必要。

继续阅读[实验协议](protocol.html)、[行为分析](behavior.html)，或原始[研究草案](../reference/docs/proposal.html)、[训练契约](../reference/docs/training.html)、[配对 gate 规范](../reference/docs/paired_validation.html)。
