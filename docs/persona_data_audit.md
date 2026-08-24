# ShopSimulator Persona 数据审计

> 生成时间：2026-08-23T04:25:44+00:00  
> 审计脚本：`scripts/extract_persona_data.py`  
> 原始文件 SHA256：`57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f`

## 1. 执行结论

已从原始商品文件中提取 **4,666** 条 persona 任务，其中 train **3,323** 条、test **1,343** 条。persona 的判定条件是同一记录同时包含非空 `user_persona` 和非空 `instruction_simple`。提取时未修改 profile 或标签内容。

**高优先级发现：当前 train 为 3,323 条，而论文报告 3,383 条，相差 -60 条。** test 的 1,343 条与论文一致。在确认官方数据 revision 前，不应从 test 挪数据补齐 train。

**阻断正式训练的问题：当前解压数据与仓库环境代码存在 schema 不匹配。** 数据中全部记录缺少顶层 `query` 和 `reason_key`，全部 instruction 缺少 `instruction_sample`；而当前环境代码会读取这些字段。需要先完成环境 adapter 或确认是否下载了与代码匹配的数据 revision。

整体上，persona/profile 主体字段完整，目标 ASIN 和商品关联良好；但 profile 中存在大量尾随逗号、空占位符等格式噪声。建议训练前保留 raw 数据，并由 prompt serializer 做确定性的非破坏性清洗，不能直接覆盖原始字段。

## 2. 来源与提取范围

| 项目 | 数值 |
|---|---:|
| 原始商品记录 | 23,421 |
| 原始 train tag | 21,962 |
| 原始 eval tag | 1,459 |
| 含 persona 的商品记录 | 4,666 |
| 含非空 instruction_simple 的任务 | 4,666 |
| 最终 persona train | 3,323 |
| 最终 persona test | 1,343 |
| Persona train 原始 item index | 1,459-4,781 |
| Persona test 原始 item index | 0-1,342 |

输出文件：

- `data/persona/persona_train.jsonl`
- `data/persona/persona_test.jsonl`
- `data/persona/manifest.json`
- `data/persona/audit.json`
- `data/persona/issues.csv`

每条 JSONL 将模型输入放在 `input.instruction` 和 `input.user_persona` 中；完整需求和目标答案只放在 `reference` 中。构造 policy prompt 时严禁把 `reference.full_instruction`、`target_asin` 或 target options 暴露给模型。

## 3. Split 与分布

### 3.1 领域分布

| Domain | Train | Test | Total |
|---|---:|---:|---:|
| appliances | 384 | 150 | 534 |
| beauty | 512 | 183 | 695 |
| clothing | 478 | 161 | 639 |
| food | 203 | 86 | 289 |
| home | 509 | 247 | 756 |
| kids | 129 | 67 | 196 |
| leisure | 414 | 179 | 593 |
| sports | 212 | 81 | 293 |
| supplies | 482 | 189 | 671 |

### 3.2 精确重复与跨 split 重叠

| 检查 | 数量 |
|---|---:|
| Train/test ASIN 重叠 | 0 |
| Train/test 简化 instruction 精确重叠 | 0 |
| Train/test 完整 instruction 精确重叠 | 0 |
| Train/test 用户 ID 重叠 | 3 |
| Train/test 完整 persona 指纹重叠 | 0 |
| Train 内简化 instruction 多余重复行 | 0 |
| Test 内简化 instruction 多余重复行 | 0 |

这里检查的是 Unicode/空白归一化后的精确重复，不等价于语义近重复检查。正式切分 `train_core/curriculum_control/dev` 前，仍建议增加 embedding 或字符 n-gram 近重复聚类。

## 4. Schema 与完整性

| 检查 | 通过数 | 通过率 |
|---|---:|---:|
| instruction ASIN 与 item ASIN 一致 | 4,666/4,666 | 100.00% |
| 所有 target options 可在商品 options 中找到 | 4,594/4,666 | 98.46% |
| target attributes 是商品 attributes 子集 | 4,611/4,666 | 98.82% |
| pricing 非空且为数值 | 4,666/4,666 | 100.00% |
| 所有 persona 必需路径均存在且非 null | 4,417/4,666 | 94.66% |

71 条 target option 关联异常中，有 63 条的目标文本包含 `?`，而商品 option 对应位置通常是 emoji 或特殊符号，疑似文本替换/编码损伤。另有 55 条任务的目标属性为空白；当前环境 `get_goals()` 会跳过空属性任务，可能导致任务数和索引整体偏移。二者都应在环境重放测试中重点处理。

### 缺失字段

机器可读的字段缺失计数位于 `audit.json`。主要 issue 计数如下：

| Issue code | Rows |
|---|---:|
| `persona_placeholder_string` | 4,666 |
| `user_id_trailing_separator` | 4,562 |
| `missing_persona_path` | 249 |
| `nested_transaction_features` | 248 |
| `target_option_not_in_catalog` | 71 |
| `missing_or_blank_target_attributes` | 55 |
| `missing_or_blank_target_options` | 1 |

## 5. Persona 质量

| 检查 | 数量 |
|---|---:|
| 唯一非空用户 ID | 4,009 |
| 用户 ID 带尾随逗号/空白 | 4,562 |
| profile 含空字符串/逗号型占位符 | 4,666 |
| `交易特征` 错误嵌套在 `行为特征` 下 | 248 |
| profile 含 `__reasoning__` | 0 |
| active hour 存在非 0-23 整数 | 0 |
| 复购率或优惠券使用率不在 [0,1] | 0 |
| 价格偏好 min > max | 0 |

profile 中的搜索历史、品牌/材质/颜色偏好有意承载被 `instruction_simple` 省略的个性化约束。审计同时计算了目标属性在 persona 中的字面覆盖率；它是任务设计特征，不应被误判为 test 泄漏。用户 ID 和地区等字段应按潜在敏感字段处理，日志与公开轨迹中建议散列或移除用户 ID。

## 6. 长度统计

长度单位为 Unicode 字符，不等同于 Qwen tokenizer token 数。正式训练前应使用冻结的 Qwen3.5 tokenizer 补做 token 审计。

| 字段 | 统计 |
|---|---|
| 简化 instruction | mean=26.9, median=26.0, P95=38.0, max=62 |
| 完整 instruction | mean=70.2, median=68.0, P95=107.0, max=189 |
| persona JSON | mean=883.1, median=863.0, P95=1006.0, max=1146 |
| instruction + persona | mean=909.9, median=890.0, P95=1033.8, max=1172 |
| 每任务目标属性数 | mean=4.6, median=4.0, P95=9.0, max=18 |
| 每任务目标 option 数 | mean=1.2, median=1.0, P95=2.0, max=4 |

## 7. 与当前环境代码的兼容性

| 环境引用字段 | 原始数据非空数 | 结论 |
|---|---:|---|
| item.query | 0/23,421 | 当前数据全部缺失时，`engine.py` 的严格索引会失败 |
| item.reason_key | 0/23,421 | persona reasoning 路径不可用 |
| instruction.instruction_sample | 0/23,421 | `goal.py` persona 分支会失败 |

在训练前应选择并记录一种修复方式：优先确认官方数据 revision；若确认当前数据就是目标版本，则写项目侧 adapter，明确使用 `instruction_simple`，并为 `query/reason_key` 定义可测试的兼容逻辑。不要直接篡改原始 JSON 来掩盖版本问题。

## 8. 建议与准入结论

当前 persona 数据可用于后续清洗和 split 设计，但**尚不应直接启动正式 SFT/RL**。建议按顺序完成：

1. 记录数据来源和 Hugging Face revision，确认 train 少 60 条的原因；
2. 修复或适配环境 schema，并对 persona train/test 各重放至少 20 条；
3. 对 `input.user_persona` 只做运行时确定性清洗，保留 raw JSONL；
4. 使用 Qwen3.5 tokenizer 统计完整 system prompt + persona + observation 的 token 分布；
5. 在 3,323 条 train 内完成分层 `train_core/curriculum_control/dev` 划分；
6. 对新划分执行 ASIN、instruction、persona 指纹和语义近重复检查；
7. 冻结 manifest 后再生成 SFT 轨迹。

## 9. 可复现命令

```powershell
python scripts/extract_persona_data.py `
  --source ShopSimulator/shop_env/data/fine_items_eval_train_all.json `
  --output-dir data/persona `
  --report docs/persona_data_audit.md
```
