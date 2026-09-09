# 实验协议与运行时

本报告覆盖冻结的 Qwen3.5-4B Iter80 实验，区分研究草案、当前实现契约与本次实际运行设置。本地环境作为普通 `ShopSimulator/` 目录随项目提供。

## 任务、观测与奖励

场景为 ShopSimulator single-turn 个性化购物。任务提供购买请求和 persona，智能体读取公开购物 observation 与当前页面可用工具，再搜索、打开商品、选择规格并购买。冻结 persona split 为 **train 3,726、validation 400、test 400**。

本次环境版本为 `shopsimulator-paper-aligned-v8`，observation 为 `shopping-observation-v7`，action protocol 为 `openai-function-tools-repair-v3`。优化目标是 strict reward（`r_strict`，也以 `reward` 暴露）。Success 表示获得完整奖励，并非仅成功抵达购买终局。属性、选项、类型、价格和个性化分量用于诊断部分匹配，精确定义见[环境契约](../reference/ShopSimulator/docs/ENVIRONMENT.html)。

一次模型生成可能没有合法工具调用。Runtime 可返回协议反馈，让模型修复，而不执行环境动作。已提交但被环境判无效的操作是另一种事件。购买动作在协议上成功终止，也可能没有满足任务奖励条件。

## 冻结设置

| 项目 | 本次运行 |
|---|---|
| Policy | Qwen3.5-4B；最终评测身份 `qwen35-4b-iter80` |
| 轮次 | 四轮，每轮 20 个 rollout step |
| Siblings | 每组八条轨迹，共享 mask |
| 生成 / 接纳 | 每 rollout step 为 32 / 16 groups |
| Global batch | 64 episodes，每 rollout step 两次优化器更新 |
| 学习率 | 两个 optimizer parameter groups 均为固定 1 × 10⁻⁶，全部 W&B updates 已核验 |
| 指导 | q 固定 0.20；rho 按贡献映射至 [0.20, 0.90] |
| 技能预算 | 最多 10 个 active chunks，每轮最多 6 个在线候选 |
| Gate | 每个 checkpoint 400 bare + 400 random masked |
| 采样 | Temperature 0.6、top_p 1.0、开启 thinking |
| 上限 | 每次生成 4,096 token；每 episode 30 模型步 |
| Seed | `send_seed=false`，任务 seed 一致不代表解码共享随机数 |

Launcher 默认四卡，trainer TP=2、rollout TP=1。这是执行配置，不是硬件无关成本声明。训练通过 slime 与 SGLang 扩展点接入；policy runtime 与通用 evaluator 共享购物契约。当前命令与拓扑约束见[训练指南](../reference/docs/training.html)。

## 三类评测

**训练生成**随策略、训练任务和辅助 mask 变化，包含随后被过滤的 groups。它适合诊断训练行为，但不是独立 held-out 性能估计。

**Validation** 在 Base、Iter20、Iter40、Iter60、Iter80 上使用固定 400 任务。Bare validation 支持纵向行为对比；四个训练后 checkpoint 的 random-mask validation 用于估计 chunk 贡献、选择下一版技能。解释验证集增益时必须考虑这种重复选择用途。

**最终 test** 比较 Base free、Base + 初始 S₀、Iter80 free、Iter80 + 最终 Sₜ。四个条件均在相同 400 任务上各有一条完成轨迹，覆盖完整。Base free 另有 21 次后来补测成功的 failed attempts，不作为额外样本。若同一任务出现重复 completed 记录，分析会拒绝而不是静默保留最新一条。

## 证据范围

公开分析严格覆盖四个完整轮次，rollout steps 1–80。组级汇总区分 20,480 条 generated episodes 与 10,240 条 trainer-admitted episodes，额外诊断 full-skill retries 不计入这些训练样本数。

初始 S₀ 来自历史冻结上下文恢复，`uses_test_outcomes=false`、`validation_refitted=false`。它不是一次重新运行、满足所有当前 provenance 要求的新 Gate A。在线 gates 保留逐任务配对表、selected banks 与 curriculum states。

行为长度方面，训练 trace 保存输出文本，但没有原始采样 token IDs。本次使用官方 Qwen tokenizer 对文本重新编码，记录文件 SHA-256，并明确标记无法核验其与历史 tokenizer hash 一致。评测 trace 使用 API `completion_tokens`。两种来源分别标注，都不替代 optimizer 实际使用的精确 token 流。

## 数据与来源追溯

分析摘录记录输入哈希。公开数值 episode / turn 表不含 persona 正文、完整对话或商品目录正文。案例只保留动作类别和成本，并公开选择规则。重建汇总图表不需要 GPU、模型 endpoint、W&B key 或原始 runs；刷新原始摘录才需要本地轨迹和 tokenizer，刷新 W&B 才需要 run 读取权限。

继续阅读[结果](results.html)、[行为](behavior.html)、[复现](reproduce.html)和[原始来源说明](../reference/docs/provenance.html)。
