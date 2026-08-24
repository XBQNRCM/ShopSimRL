#!/usr/bin/env python3
"""Extract ShopSimulator persona tasks and produce a reproducible data audit.

The source file stores product records. A record is a persona task when it has
both a non-empty ``user_persona`` dictionary and a non-empty
``instructions[*].instruction_simple`` field. The full instruction is retained
under ``reference`` for reward/debugging only and must never be exposed to the
policy as input.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "shopsimrl.persona_task.v1"
EXPECTED_PERSONA_COUNTS = {"train": 3383, "test": 1343}
REQUIRED_ITEM_FIELDS = (
    "asin",
    "tag",
    "domain_zh",
    "domain_en_short",
    "domain_en_long",
    "title",
    "category",
    "attribute",
    "customization_options",
    "pricing",
    "instructions",
)
REQUIRED_INSTRUCTION_FIELDS = (
    "asin",
    "instruction",
    "instruction_simple",
    "instruction_options",
    "attributes",
)
EXPECTED_PERSONA_PATHS = (
    "用户ID",
    "注册时间",
    "注册渠道",
    "地区信息.省份",
    "地区信息.城市",
    "地区信息.区县",
    "地区信息.时区",
    "人口属性.性别",
    "人口属性.年龄段",
    "人口属性.消费等级",
    "人口属性.会员等级",
    "行为特征.日均浏览时长",
    "行为特征.近7天访问次数",
    "行为特征.活跃时间段",
    "行为特征.常用设备",
    "行为特征.最近14天搜索关键词",
    "行为特征.最近14天收藏商品",
    "行为特征.最近14天加购商品",
    "行为特征.最近14天关注店铺",
    "交易特征.近30天消费金额",
    "交易特征.近90天订单数",
    "交易特征.平均客单价",
    "交易特征.复购率",
    "交易特征.支付方式偏好",
    "交易特征.优惠券使用率",
    "交易特征.是否促销敏感",
    "兴趣偏好.类目偏好",
    "兴趣偏好.品牌偏好",
    "兴趣偏好.商品属性偏好",
    "用户标签",
    "最后更新时间",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("ShopSimulator/shop_env/data/fine_items_eval_train_all.json"),
        help="Path to the decompressed ShopSimulator JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/persona"),
        help="Directory for extracted JSONL files and machine-readable audit files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/persona_data_audit.md"),
        help="Markdown audit report path.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", text)


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def get_path(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p95": percentile(numbers, 0.95),
        "max": max(numbers),
    }


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda pair: str(pair[0]))}


def duplicate_summary(values: list[str]) -> dict[str, int]:
    counts = Counter(value for value in values if value)
    duplicate_groups = [count for count in counts.values() if count > 1]
    return {
        "nonempty_values": sum(counts.values()),
        "unique_values": len(counts),
        "duplicate_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(duplicate_groups),
        "excess_duplicate_rows": sum(count - 1 for count in duplicate_groups),
    }


def nested_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_strings(child)


def all_catalog_option_values(item: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    options = item.get("customization_options")
    if not isinstance(options, dict):
        return values
    for contents in options.values():
        if not isinstance(contents, list):
            continue
        for option in contents:
            if isinstance(option, dict) and is_nonempty(option.get("value")):
                values.add(normalize_text(option["value"]))
    return values


def split_from_tag(tag: Any) -> str | None:
    normalized = normalize_text(tag)
    if normalized == "train":
        return "train"
    if normalized in {"eval", "test"}:
        return "test"
    return None


def make_task_record(
    item: dict[str, Any],
    instruction: dict[str, Any],
    split: str,
    persona_index: int,
    item_index: int,
    instruction_index: int,
) -> dict[str, Any]:
    asin = str(item.get("asin", ""))
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": f"persona-{split}-{persona_index:05d}-{asin}",
        "split": split,
        "persona_index": persona_index,
        "source": {
            "tag": item.get("tag"),
            "item_index": item_index,
            "instruction_index": instruction_index,
            "worker_id": instruction.get("worker_id"),
        },
        "asin": asin,
        "domain": {
            "zh": item.get("domain_zh"),
            "en_short": item.get("domain_en_short"),
            "en_long": item.get("domain_en_long"),
        },
        "product": {
            "title": item.get("title"),
            "sub_title": item.get("sub_title"),
            "shop_name": item.get("shop_name"),
            "category": item.get("category"),
            "pricing": item.get("pricing"),
        },
        "input": {
            "instruction": instruction.get("instruction_simple"),
            "user_persona": item.get("user_persona"),
        },
        "reference": {
            "full_instruction": instruction.get("instruction"),
            "target_asin": instruction.get("asin"),
            "target_attributes": instruction.get("attributes"),
            "target_options": instruction.get("instruction_options"),
            "raw_options": instruction.get("options"),
        },
    }


def add_issue(
    issues: list[dict[str, Any]],
    task_id: str,
    split: str,
    severity: str,
    code: str,
    detail: str,
) -> None:
    issues.append(
        {
            "task_id": task_id,
            "split": split,
            "severity": severity,
            "code": code,
            "detail": detail,
        }
    )


def build_report(audit: dict[str, Any]) -> str:
    source = audit["source"]
    extraction = audit["extraction"]
    integrity = audit["referential_integrity"]
    leakage = audit["duplicates_and_split_overlap"]
    profile = audit["persona_quality"]
    lengths = audit["lengths"]
    compatibility = audit["code_compatibility"]
    issue_counts = audit["issues"]["by_code"]

    def count(split: str) -> int:
        return extraction["counts_by_split"].get(split, 0)

    def pct(numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "n/a"
        return f"{100 * numerator / denominator:.2f}%"

    def fmt_stats(stats: dict[str, Any]) -> str:
        if not stats["count"]:
            return "n/a"
        return (
            f"mean={stats['mean']:.1f}, median={stats['median']:.1f}, "
            f"P95={stats['p95']:.1f}, max={stats['max']:.0f}"
        )

    total = extraction["total_persona_tasks"]
    missing_paths = profile["missing_required_path_rows"]
    all_options_match = integrity["target_options_all_in_catalog"]
    asin_match = integrity["instruction_asin_matches_item"]

    lines = [
        "# ShopSimulator Persona 数据审计",
        "",
        f"> 生成时间：{audit['generated_at_utc']}  ",
        f"> 审计脚本：`scripts/extract_persona_data.py`  ",
        f"> 原始文件 SHA256：`{source['sha256']}`",
        "",
        "## 1. 执行结论",
        "",
        f"已从原始商品文件中提取 **{total:,}** 条 persona 任务，其中 train **{count('train'):,}** 条、test **{count('test'):,}** 条。persona 的判定条件是同一记录同时包含非空 `user_persona` 和非空 `instruction_simple`。提取时未修改 profile 或标签内容。",
        "",
    ]

    if count("train") != EXPECTED_PERSONA_COUNTS["train"]:
        delta = count("train") - EXPECTED_PERSONA_COUNTS["train"]
        lines.extend(
            [
                f"**高优先级发现：当前 train 为 {count('train'):,} 条，而论文报告 {EXPECTED_PERSONA_COUNTS['train']:,} 条，相差 {delta:+,} 条。** test 的 {count('test'):,} 条与论文一致。在确认官方数据 revision 前，不应从 test 挪数据补齐 train。",
                "",
            ]
        )

    lines.extend(
        [
            "**阻断正式训练的问题：当前解压数据与仓库环境代码存在 schema 不匹配。** 数据中全部记录缺少顶层 `query` 和 `reason_key`，全部 instruction 缺少 `instruction_sample`；而当前环境代码会读取这些字段。需要先完成环境 adapter 或确认是否下载了与代码匹配的数据 revision。",
            "",
            "整体上，persona/profile 主体字段完整，目标 ASIN 和商品关联良好；但 profile 中存在大量尾随逗号、空占位符等格式噪声。建议训练前保留 raw 数据，并由 prompt serializer 做确定性的非破坏性清洗，不能直接覆盖原始字段。",
            "",
            "## 2. 来源与提取范围",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
            f"| 原始商品记录 | {source['item_count']:,} |",
            f"| 原始 train tag | {source['tag_counts'].get('train', 0):,} |",
            f"| 原始 eval tag | {source['tag_counts'].get('eval', 0):,} |",
            f"| 含 persona 的商品记录 | {source['items_with_nonempty_persona']:,} |",
            f"| 含非空 instruction_simple 的任务 | {source['instructions_with_nonempty_simple']:,} |",
            f"| 最终 persona train | {count('train'):,} |",
            f"| 最终 persona test | {count('test'):,} |",
            f"| Persona train 原始 item index | {extraction['source_item_index_range']['train'][0]:,}-{extraction['source_item_index_range']['train'][1]:,} |",
            f"| Persona test 原始 item index | {extraction['source_item_index_range']['test'][0]:,}-{extraction['source_item_index_range']['test'][1]:,} |",
            "",
            "输出文件：",
            "",
            "- `data/persona/persona_train.jsonl`",
            "- `data/persona/persona_test.jsonl`",
            "- `data/persona/manifest.json`",
            "- `data/persona/audit.json`",
            "- `data/persona/issues.csv`",
            "",
            "每条 JSONL 将模型输入放在 `input.instruction` 和 `input.user_persona` 中；完整需求和目标答案只放在 `reference` 中。构造 policy prompt 时严禁把 `reference.full_instruction`、`target_asin` 或 target options 暴露给模型。",
            "",
            "## 3. Split 与分布",
            "",
            "### 3.1 领域分布",
            "",
            "| Domain | Train | Test | Total |",
            "|---|---:|---:|---:|",
        ]
    )
    domains = sorted(
        set(extraction["domain_counts"].get("train", {}))
        | set(extraction["domain_counts"].get("test", {}))
    )
    for domain in domains:
        train_value = extraction["domain_counts"].get("train", {}).get(domain, 0)
        test_value = extraction["domain_counts"].get("test", {}).get(domain, 0)
        lines.append(f"| {domain} | {train_value:,} | {test_value:,} | {train_value + test_value:,} |")

    lines.extend(
        [
            "",
            "### 3.2 精确重复与跨 split 重叠",
            "",
            "| 检查 | 数量 |",
            "|---|---:|",
            f"| Train/test ASIN 重叠 | {leakage['train_test_asin_overlap']:,} |",
            f"| Train/test 简化 instruction 精确重叠 | {leakage['train_test_simple_instruction_overlap']:,} |",
            f"| Train/test 完整 instruction 精确重叠 | {leakage['train_test_full_instruction_overlap']:,} |",
            f"| Train/test 用户 ID 重叠 | {leakage['train_test_user_id_overlap']:,} |",
            f"| Train/test 完整 persona 指纹重叠 | {leakage['train_test_persona_fingerprint_overlap']:,} |",
            f"| Train 内简化 instruction 多余重复行 | {leakage['within_split']['train']['simple_instruction']['excess_duplicate_rows']:,} |",
            f"| Test 内简化 instruction 多余重复行 | {leakage['within_split']['test']['simple_instruction']['excess_duplicate_rows']:,} |",
            "",
            "这里检查的是 Unicode/空白归一化后的精确重复，不等价于语义近重复检查。正式切分 `train_core/curriculum_control/dev` 前，仍建议增加 embedding 或字符 n-gram 近重复聚类。",
            "",
            "## 4. Schema 与完整性",
            "",
            "| 检查 | 通过数 | 通过率 |",
            "|---|---:|---:|",
            f"| instruction ASIN 与 item ASIN 一致 | {asin_match:,}/{total:,} | {pct(asin_match, total)} |",
            f"| 所有 target options 可在商品 options 中找到 | {all_options_match:,}/{total:,} | {pct(all_options_match, total)} |",
            f"| target attributes 是商品 attributes 子集 | {integrity['target_attributes_subset_of_product']:,}/{total:,} | {pct(integrity['target_attributes_subset_of_product'], total)} |",
            f"| pricing 非空且为数值 | {integrity['valid_numeric_pricing']:,}/{total:,} | {pct(integrity['valid_numeric_pricing'], total)} |",
            f"| 所有 persona 必需路径均存在且非 null | {total - missing_paths:,}/{total:,} | {pct(total - missing_paths, total)} |",
            "",
            f"{integrity['target_option_not_in_catalog']:,} 条 target option 关联异常中，有 {integrity['target_option_mismatch_contains_question_mark']:,} 条的目标文本包含 `?`，而商品 option 对应位置通常是 emoji 或特殊符号，疑似文本替换/编码损伤。另有 {integrity['missing_or_blank_target_attributes']:,} 条任务的目标属性为空白；当前环境 `get_goals()` 会跳过空属性任务，可能导致任务数和索引整体偏移。二者都应在环境重放测试中重点处理。",
            "",
            "### 缺失字段",
            "",
            "机器可读的字段缺失计数位于 `audit.json`。主要 issue 计数如下：",
            "",
            "| Issue code | Rows |",
            "|---|---:|",
        ]
    )
    if issue_counts:
        for code, value in sorted(issue_counts.items(), key=lambda pair: (-pair[1], pair[0])):
            lines.append(f"| `{code}` | {value:,} |")
    else:
        lines.append("| 无 | 0 |")

    lines.extend(
        [
            "",
            "## 5. Persona 质量",
            "",
            "| 检查 | 数量 |",
            "|---|---:|",
            f"| 唯一非空用户 ID | {profile['unique_nonempty_user_ids']:,} |",
            f"| 用户 ID 带尾随逗号/空白 | {profile['user_id_trailing_separator_rows']:,} |",
            f"| profile 含空字符串/逗号型占位符 | {profile['rows_with_placeholder_strings']:,} |",
            f"| `交易特征` 错误嵌套在 `行为特征` 下 | {profile['rows_with_nested_transaction_features']:,} |",
            f"| profile 含 `__reasoning__` | {profile['rows_with_reasoning_field']:,} |",
            f"| active hour 存在非 0-23 整数 | {profile['rows_with_invalid_active_hours']:,} |",
            f"| 复购率或优惠券使用率不在 [0,1] | {profile['rows_with_invalid_rates']:,} |",
            f"| 价格偏好 min > max | {profile['rows_with_invalid_preference_price_range']:,} |",
            "",
            "profile 中的搜索历史、品牌/材质/颜色偏好有意承载被 `instruction_simple` 省略的个性化约束。审计同时计算了目标属性在 persona 中的字面覆盖率；它是任务设计特征，不应被误判为 test 泄漏。用户 ID 和地区等字段应按潜在敏感字段处理，日志与公开轨迹中建议散列或移除用户 ID。",
            "",
            "## 6. 长度统计",
            "",
            "长度单位为 Unicode 字符，不等同于 Qwen tokenizer token 数。正式训练前应使用冻结的 Qwen3.5 tokenizer 补做 token 审计。",
            "",
            "| 字段 | 统计 |",
            "|---|---|",
            f"| 简化 instruction | {fmt_stats(lengths['simple_instruction_chars'])} |",
            f"| 完整 instruction | {fmt_stats(lengths['full_instruction_chars'])} |",
            f"| persona JSON | {fmt_stats(lengths['persona_json_chars'])} |",
            f"| instruction + persona | {fmt_stats(lengths['input_chars'])} |",
            f"| 每任务目标属性数 | {fmt_stats(lengths['target_attribute_count'])} |",
            f"| 每任务目标 option 数 | {fmt_stats(lengths['target_option_count'])} |",
            "",
            "## 7. 与当前环境代码的兼容性",
            "",
            "| 环境引用字段 | 原始数据非空数 | 结论 |",
            "|---|---:|---|",
            f"| item.query | {compatibility['nonempty_item_query']:,}/{source['item_count']:,} | 当前数据全部缺失时，`engine.py` 的严格索引会失败 |",
            f"| item.reason_key | {compatibility['nonempty_item_reason_key']:,}/{source['item_count']:,} | persona reasoning 路径不可用 |",
            f"| instruction.instruction_sample | {compatibility['nonempty_instruction_sample']:,}/{source['instruction_count']:,} | `goal.py` persona 分支会失败 |",
            "",
            "在训练前应选择并记录一种修复方式：优先确认官方数据 revision；若确认当前数据就是目标版本，则写项目侧 adapter，明确使用 `instruction_simple`，并为 `query/reason_key` 定义可测试的兼容逻辑。不要直接篡改原始 JSON 来掩盖版本问题。",
            "",
            "## 8. 建议与准入结论",
            "",
            "当前 persona 数据可用于后续清洗和 split 设计，但**尚不应直接启动正式 SFT/RL**。建议按顺序完成：",
            "",
            "1. 记录数据来源和 Hugging Face revision，确认 train 少 60 条的原因；",
            "2. 修复或适配环境 schema，并对 persona train/test 各重放至少 20 条；",
            "3. 对 `input.user_persona` 只做运行时确定性清洗，保留 raw JSONL；",
            "4. 使用 Qwen3.5 tokenizer 统计完整 system prompt + persona + observation 的 token 分布；",
            "5. 在 3,323 条 train 内完成分层 `train_core/curriculum_control/dev` 划分；",
            "6. 对新划分执行 ASIN、instruction、persona 指纹和语义近重复检查；",
            "7. 冻结 manifest 后再生成 SFT 轨迹。",
            "",
            "## 9. 可复现命令",
            "",
            "```powershell",
            "python scripts/extract_persona_data.py `",
            "  --source ShopSimulator/shop_env/data/fine_items_eval_train_all.json `",
            "  --output-dir data/persona `",
            "  --report docs/persona_data_audit.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise TypeError(f"Expected a JSON list, got {type(raw).__name__}")

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    issues: list[dict[str, Any]] = []

    tag_counts: Counter[str] = Counter()
    item_key_counts: Counter[str] = Counter()
    instruction_key_counts: Counter[str] = Counter()
    missing_item_fields: Counter[str] = Counter()
    missing_instruction_fields: Counter[str] = Counter()
    domain_counts: dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}
    worker_id_counts: dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}
    source_asin_counts: Counter[str] = Counter()

    items_with_nonempty_persona = 0
    instructions_with_nonempty_simple = 0
    persona_without_simple = 0
    simple_without_persona = 0
    instruction_count = 0
    nonempty_item_query = 0
    nonempty_item_reason_key = 0
    nonempty_instruction_sample = 0

    integrity = Counter()
    missing_persona_paths: Counter[str] = Counter()
    persona_metrics = Counter()
    persona_schema_counts: Counter[str] = Counter()
    categorical: dict[str, Counter[str]] = defaultdict(Counter)
    length_values: dict[str, list[float]] = defaultdict(list)
    target_attribute_persona_coverage: list[float] = []
    target_attribute_simple_coverage: list[float] = []

    normalized_values: dict[str, dict[str, list[str]]] = {
        "train": defaultdict(list),
        "test": defaultdict(list),
    }

    for item_index, item in enumerate(raw):
        if not isinstance(item, dict):
            add_issue(issues, f"source-item-{item_index}", "unknown", "error", "item_not_object", type(item).__name__)
            continue

        tag = normalize_text(item.get("tag")) or "<missing>"
        tag_counts[tag] += 1
        split = split_from_tag(tag)
        item_key_counts["|".join(sorted(item.keys()))] += 1
        asin = normalize_text(item.get("asin"))
        if asin:
            source_asin_counts[asin] += 1
        for field in REQUIRED_ITEM_FIELDS:
            if not is_nonempty(item.get(field)):
                missing_item_fields[field] += 1
        if is_nonempty(item.get("query")):
            nonempty_item_query += 1
        if is_nonempty(item.get("reason_key")):
            nonempty_item_reason_key += 1

        persona = item.get("user_persona")
        has_persona = isinstance(persona, dict) and bool(persona)
        if has_persona:
            items_with_nonempty_persona += 1

        instructions = item.get("instructions")
        if not isinstance(instructions, list):
            continue
        for instruction_index, instruction in enumerate(instructions):
            instruction_count += 1
            if not isinstance(instruction, dict):
                add_issue(
                    issues,
                    f"source-item-{item_index}-instruction-{instruction_index}",
                    split or "unknown",
                    "error",
                    "instruction_not_object",
                    type(instruction).__name__,
                )
                continue
            instruction_key_counts["|".join(sorted(instruction.keys()))] += 1
            for field in REQUIRED_INSTRUCTION_FIELDS:
                if not is_nonempty(instruction.get(field)):
                    missing_instruction_fields[field] += 1
            if is_nonempty(instruction.get("instruction_sample")):
                nonempty_instruction_sample += 1

            has_simple = is_nonempty(instruction.get("instruction_simple"))
            if has_simple:
                instructions_with_nonempty_simple += 1
            if has_persona and not has_simple:
                persona_without_simple += 1
            if has_simple and not has_persona:
                simple_without_persona += 1
            if not (has_persona and has_simple):
                continue
            if split is None:
                add_issue(
                    issues,
                    f"source-item-{item_index}-instruction-{instruction_index}",
                    "unknown",
                    "error",
                    "unknown_split_tag",
                    str(item.get("tag")),
                )
                continue

            persona_index = len(records_by_split[split])
            record = make_task_record(
                item,
                instruction,
                split,
                persona_index,
                item_index,
                instruction_index,
            )
            records_by_split[split].append(record)
            task_id = record["task_id"]

            domain = normalize_text(item.get("domain_en_short")) or normalize_text(item.get("domain_zh")) or "<missing>"
            domain_counts[split][domain] += 1
            worker_id_counts[split][normalize_text(instruction.get("worker_id")) or "<missing>"] += 1

            simple = normalize_text(instruction.get("instruction_simple"))
            full = normalize_text(instruction.get("instruction"))
            user_id = normalize_text(persona.get("用户ID"))
            persona_fingerprint = stable_hash(persona)
            normalized_values[split]["asin"].append(asin)
            normalized_values[split]["simple_instruction"].append(simple)
            normalized_values[split]["full_instruction"].append(full)
            normalized_values[split]["user_id"].append(user_id)
            normalized_values[split]["persona_fingerprint"].append(persona_fingerprint)

            persona_json = json.dumps(persona, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            persona_schema_counts["|".join(sorted(persona.keys()))] += 1
            length_values["simple_instruction_chars"].append(len(str(instruction.get("instruction_simple", ""))))
            length_values["full_instruction_chars"].append(len(str(instruction.get("instruction", ""))))
            length_values["persona_json_chars"].append(len(persona_json))
            length_values["input_chars"].append(len(str(instruction.get("instruction_simple", ""))) + len(persona_json))

            target_attributes = instruction.get("attributes") if isinstance(instruction.get("attributes"), list) else []
            target_options = (
                instruction.get("instruction_options")
                if isinstance(instruction.get("instruction_options"), list)
                else []
            )
            length_values["target_attribute_count"].append(len(target_attributes))
            length_values["target_option_count"].append(len(target_options))

            item_asin = normalize_text(item.get("asin"))
            instruction_asin = normalize_text(instruction.get("asin"))
            if item_asin and item_asin == instruction_asin:
                integrity["instruction_asin_matches_item"] += 1
            else:
                add_issue(issues, task_id, split, "error", "instruction_asin_mismatch", f"{item_asin!r} != {instruction_asin!r}")

            catalog_options = all_catalog_option_values(item)
            normalized_targets = {normalize_text(option) for option in target_options if is_nonempty(option)}
            if not normalized_targets:
                add_issue(issues, task_id, split, "warning", "missing_or_blank_target_options", repr(target_options))
            elif normalized_targets.issubset(catalog_options):
                integrity["target_options_all_in_catalog"] += 1
            else:
                missing = sorted(normalized_targets - catalog_options)
                add_issue(issues, task_id, split, "warning", "target_option_not_in_catalog", json.dumps(missing, ensure_ascii=False))

            product_attributes = {normalize_text(value) for value in (item.get("attribute") or []) if is_nonempty(value)}
            normalized_attributes = {normalize_text(value) for value in target_attributes if is_nonempty(value)}
            if not normalized_attributes:
                add_issue(issues, task_id, split, "warning", "missing_or_blank_target_attributes", repr(target_attributes))
            elif normalized_attributes.issubset(product_attributes):
                integrity["target_attributes_subset_of_product"] += 1
            else:
                missing = sorted(normalized_attributes - product_attributes)
                add_issue(issues, task_id, split, "warning", "target_attribute_not_in_product", json.dumps(missing, ensure_ascii=False))

            pricing = item.get("pricing")
            if (
                isinstance(pricing, list)
                and bool(pricing)
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in pricing)
            ):
                integrity["valid_numeric_pricing"] += 1
            else:
                add_issue(issues, task_id, split, "warning", "invalid_pricing", repr(pricing))

            # Empty lists are valid for fields such as recent favorites/cart items.
            # Count a required path as missing only when it is absent or explicitly null.
            missing_for_row = [path for path in EXPECTED_PERSONA_PATHS if get_path(persona, path) is None]
            if missing_for_row:
                persona_metrics["missing_required_path_rows"] += 1
                for path in missing_for_row:
                    missing_persona_paths[path] += 1
                add_issue(issues, task_id, split, "warning", "missing_persona_path", "; ".join(missing_for_row))

            raw_user_id = persona.get("用户ID")
            if isinstance(raw_user_id, str) and re.search(r"[,，\s]+$", raw_user_id):
                persona_metrics["user_id_trailing_separator_rows"] += 1
                add_issue(issues, task_id, split, "info", "user_id_trailing_separator", repr(raw_user_id))

            placeholder_values = []
            for value in nested_strings(persona):
                compact = unicodedata.normalize("NFKC", value).strip()
                if compact in {"", ",", "，", ", ,", "， ，"}:
                    placeholder_values.append(value)
            if placeholder_values:
                persona_metrics["rows_with_placeholder_strings"] += 1
                add_issue(
                    issues,
                    task_id,
                    split,
                    "info",
                    "persona_placeholder_string",
                    f"count={len(placeholder_values)}",
                )

            if "__reasoning__" in persona:
                persona_metrics["rows_with_reasoning_field"] += 1

            active_hours = get_path(persona, "行为特征.活跃时间段")
            if not (
                isinstance(active_hours, list)
                and all(isinstance(hour, int) and not isinstance(hour, bool) and 0 <= hour <= 23 for hour in active_hours)
            ):
                persona_metrics["rows_with_invalid_active_hours"] += 1
                add_issue(issues, task_id, split, "warning", "invalid_active_hours", repr(active_hours))

            transaction_features = persona.get("交易特征")
            nested_transaction_features = get_path(persona, "行为特征.交易特征")
            if not isinstance(transaction_features, dict) and isinstance(nested_transaction_features, dict):
                persona_metrics["rows_with_nested_transaction_features"] += 1
                add_issue(
                    issues,
                    task_id,
                    split,
                    "warning",
                    "nested_transaction_features",
                    "交易特征 is nested under 行为特征",
                )
                transaction_features = nested_transaction_features
            repurchase_rate = transaction_features.get("复购率") if isinstance(transaction_features, dict) else None
            coupon_rate = transaction_features.get("优惠券使用率") if isinstance(transaction_features, dict) else None
            if not all(isinstance(rate, (int, float)) and not isinstance(rate, bool) and 0 <= rate <= 1 for rate in (repurchase_rate, coupon_rate)):
                persona_metrics["rows_with_invalid_rates"] += 1
                add_issue(issues, task_id, split, "warning", "invalid_rate", f"repurchase={repurchase_rate!r}, coupon={coupon_rate!r}")

            preference_range = get_path(persona, "兴趣偏好.商品属性偏好.价格区间")
            valid_range = (
                isinstance(preference_range, dict)
                and isinstance(preference_range.get("最小值"), (int, float))
                and isinstance(preference_range.get("最大值"), (int, float))
                and preference_range["最小值"] <= preference_range["最大值"]
            )
            if not valid_range:
                persona_metrics["rows_with_invalid_preference_price_range"] += 1
                add_issue(issues, task_id, split, "warning", "invalid_preference_price_range", repr(preference_range))

            categorical["gender"][normalize_text(get_path(persona, "人口属性.性别")) or "<missing>"] += 1
            categorical["age_group"][normalize_text(get_path(persona, "人口属性.年龄段")) or "<missing>"] += 1
            categorical["consumption_level"][normalize_text(get_path(persona, "人口属性.消费等级")) or "<missing>"] += 1
            categorical["membership_level"][normalize_text(get_path(persona, "人口属性.会员等级")) or "<missing>"] += 1
            categorical["province"][normalize_text(get_path(persona, "地区信息.省份")) or "<missing>"] += 1

            normalized_persona = normalize_text(persona_json)
            simple_text = normalize_text(instruction.get("instruction_simple"))
            if normalized_attributes:
                persona_hits = sum(attribute in normalized_persona for attribute in normalized_attributes)
                simple_hits = sum(attribute in simple_text for attribute in normalized_attributes)
                target_attribute_persona_coverage.append(persona_hits / len(normalized_attributes))
                target_attribute_simple_coverage.append(simple_hits / len(normalized_attributes))

    total_persona_tasks = sum(len(records) for records in records_by_split.values())
    train_values = normalized_values["train"]
    test_values = normalized_values["test"]

    within_split: dict[str, Any] = {}
    for split in ("train", "test"):
        within_split[split] = {
            key: duplicate_summary(normalized_values[split][key])
            for key in ("asin", "simple_instruction", "full_instruction", "user_id", "persona_fingerprint")
        }

    overlaps = {}
    for key in ("asin", "simple_instruction", "full_instruction", "user_id", "persona_fingerprint"):
        overlaps[key] = len(
            {value for value in train_values[key] if value}
            & {value for value in test_values[key] if value}
        )

    issue_code_counts = Counter(issue["code"] for issue in issues)
    issue_severity_counts = Counter(issue["severity"] for issue in issues)
    persona_user_ids = [
        value
        for split in ("train", "test")
        for value in normalized_values[split]["user_id"]
        if value
    ]

    audit: dict[str, Any] = {
        "schema_version": "shopsimrl.persona_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "item_count": len(raw),
            "instruction_count": instruction_count,
            "tag_counts": counter_dict(tag_counts),
            "items_with_nonempty_persona": items_with_nonempty_persona,
            "instructions_with_nonempty_simple": instructions_with_nonempty_simple,
            "persona_without_simple": persona_without_simple,
            "simple_without_persona": simple_without_persona,
            "duplicate_source_asins": sum(1 for value in source_asin_counts.values() if value > 1),
            "missing_item_fields": counter_dict(missing_item_fields),
            "missing_instruction_fields": counter_dict(missing_instruction_fields),
            "item_schema_variants": counter_dict(item_key_counts),
            "instruction_schema_variants": counter_dict(instruction_key_counts),
        },
        "extraction": {
            "criteria": "non-empty user_persona dict AND non-empty instruction_simple",
            "total_persona_tasks": total_persona_tasks,
            "counts_by_split": {split: len(records) for split, records in records_by_split.items()},
            "source_item_index_range": {
                split: [
                    min(record["source"]["item_index"] for record in records),
                    max(record["source"]["item_index"] for record in records),
                ]
                for split, records in records_by_split.items()
                if records
            },
            "expected_counts_from_paper": EXPECTED_PERSONA_COUNTS,
            "count_delta_vs_paper": {
                split: len(records_by_split[split]) - EXPECTED_PERSONA_COUNTS[split]
                for split in ("train", "test")
            },
            "domain_counts": {split: counter_dict(counter) for split, counter in domain_counts.items()},
            "worker_id_counts": {split: counter_dict(counter) for split, counter in worker_id_counts.items()},
        },
        "referential_integrity": {
            "instruction_asin_matches_item": integrity["instruction_asin_matches_item"],
            "target_options_all_in_catalog": integrity["target_options_all_in_catalog"],
            "target_attributes_subset_of_product": integrity["target_attributes_subset_of_product"],
            "valid_numeric_pricing": integrity["valid_numeric_pricing"],
            "missing_or_blank_target_options": issue_code_counts["missing_or_blank_target_options"],
            "target_option_not_in_catalog": issue_code_counts["target_option_not_in_catalog"],
            "target_option_mismatch_contains_question_mark": sum(
                1
                for issue in issues
                if issue["code"] == "target_option_not_in_catalog" and "?" in issue["detail"]
            ),
            "missing_or_blank_target_attributes": issue_code_counts["missing_or_blank_target_attributes"],
            "target_attribute_not_in_product": issue_code_counts["target_attribute_not_in_product"],
        },
        "duplicates_and_split_overlap": {
            "within_split": within_split,
            "train_test_asin_overlap": overlaps["asin"],
            "train_test_simple_instruction_overlap": overlaps["simple_instruction"],
            "train_test_full_instruction_overlap": overlaps["full_instruction"],
            "train_test_user_id_overlap": overlaps["user_id"],
            "train_test_persona_fingerprint_overlap": overlaps["persona_fingerprint"],
            "normalization": "Unicode NFKC + strip + lowercase + collapse whitespace",
        },
        "persona_quality": {
            "unique_nonempty_user_ids": len(set(persona_user_ids)),
            "missing_required_path_rows": persona_metrics["missing_required_path_rows"],
            "missing_required_path_counts": counter_dict(missing_persona_paths),
            "user_id_trailing_separator_rows": persona_metrics["user_id_trailing_separator_rows"],
            "rows_with_placeholder_strings": persona_metrics["rows_with_placeholder_strings"],
            "rows_with_nested_transaction_features": persona_metrics["rows_with_nested_transaction_features"],
            "rows_with_reasoning_field": persona_metrics["rows_with_reasoning_field"],
            "rows_with_invalid_active_hours": persona_metrics["rows_with_invalid_active_hours"],
            "rows_with_invalid_rates": persona_metrics["rows_with_invalid_rates"],
            "rows_with_invalid_preference_price_range": persona_metrics["rows_with_invalid_preference_price_range"],
            "categorical_distributions": {key: counter_dict(value) for key, value in categorical.items()},
            "top_level_schema_variants": counter_dict(persona_schema_counts),
            "target_attribute_literal_coverage_in_persona": describe(target_attribute_persona_coverage),
            "target_attribute_literal_coverage_in_simple_instruction": describe(target_attribute_simple_coverage),
        },
        "lengths": {key: describe(value) for key, value in length_values.items()},
        "code_compatibility": {
            "nonempty_item_query": nonempty_item_query,
            "nonempty_item_reason_key": nonempty_item_reason_key,
            "nonempty_instruction_sample": nonempty_instruction_sample,
            "risk": (
                "Current ShopSimulator engine code references item['query'], item['reason_key'], "
                "and instruction['instruction_sample'], but the extracted source does not provide them."
            ),
        },
        "issues": {
            "total_rows": len(issues),
            "by_severity": counter_dict(issue_severity_counts),
            "by_code": counter_dict(issue_code_counts),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_hashes: dict[str, str] = {}
    for split, records in records_by_split.items():
        output_path = output_dir / f"persona_{split}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
        output_hashes[output_path.name] = sha256_file(output_path)

    issues_path = output_dir / "issues.csv"
    with issues_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id", "split", "severity", "code", "detail"))
        writer.writeheader()
        writer.writerows(issues)
    output_hashes[issues_path.name] = sha256_file(issues_path)

    audit_path = output_dir / "audit.json"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    output_hashes[audit_path.name] = sha256_file(audit_path)

    manifest = {
        "schema_version": "shopsimrl.persona_manifest.v1",
        "generated_at_utc": audit["generated_at_utc"],
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": audit["source"]["sha256"],
        },
        "extraction": {
            "script": str(Path(__file__).resolve()),
            "criteria": audit["extraction"]["criteria"],
            "record_schema": SCHEMA_VERSION,
            "counts_by_split": audit["extraction"]["counts_by_split"],
        },
        "outputs": output_hashes,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    report_path.write_text(build_report(audit), encoding="utf-8", newline="\n")

    print(json.dumps({
        "source": str(source_path),
        "output_dir": str(output_dir),
        "report": str(report_path),
        "counts": audit["extraction"]["counts_by_split"],
        "issues": audit["issues"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
