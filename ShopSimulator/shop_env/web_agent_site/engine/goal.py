"""Task goals and the paper-aligned ShopSimulator reward."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import math
import re

from web_agent_site.engine.budget import explicit_budget_from_instruction
from web_agent_site.engine.variant_price import PASS, resolve_variant_price


PAPER_REWARD_VERSION = "shopsimulator-paper-reward-v2"
PRIMARY_REWARD = "r_strict"
_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("zh_core_web_sm")
    return _NLP


def _fuzzy_ratio(left: object, right: object) -> int:
    """Dependency-free equivalent of fuzzywuzzy's token-set ratio."""
    def tokens(value: object) -> set[str]:
        normalized = re.sub(r"\W+", " ", str(value).casefold(), flags=re.UNICODE)
        return set(normalized.split())

    left_tokens, right_tokens = tokens(left), tokens(right)
    intersection = left_tokens & right_tokens
    left_only = left_tokens - intersection
    right_only = right_tokens - intersection

    common = " ".join(sorted(intersection)).strip()
    combined_left = " ".join(sorted(intersection | left_only)).strip()
    combined_right = " ".join(sorted(intersection | right_only)).strip()

    def ratio(first: str, second: str) -> int:
        return round(100 * SequenceMatcher(None, first, second).ratio())

    if not common:
        return ratio(combined_left, combined_right)
    return max(
        ratio(common, combined_left),
        ratio(common, combined_right),
        ratio(combined_left, combined_right),
    )


def get_goals(all_products, product_prices=None, if_persona=False):
    """Build private evaluation goals from the released task annotations.

    ``product_prices`` and ``if_persona`` remain accepted for compatibility,
    but goal construction is deterministic and independent of presentation
    mode.  In particular, it never invents a budget from the Gold price.
    """
    del product_prices, if_persona
    goals = []
    attribute_counts = defaultdict(int)
    skipped_missing_instructions = 0
    skipped_missing_attributes = 0
    seen_task_ids = set()

    for item in all_products:
        instructions = item.get("instructions")
        if not instructions:
            skipped_missing_instructions += 1
            continue
        asin = item["asin"]
        for instruction in instructions:
            attributes = list(instruction.get("attributes") or [])
            if not attributes:
                skipped_missing_attributes += 1
                continue
            raw_persona = item.get("user_persona")
            persona = raw_persona.copy() if isinstance(raw_persona, dict) else {}
            complete_instruction = instruction["instruction"]
            simple_instruction = (
                instruction.get("instruction_simple") or complete_instruction
            )
            raw_upper = instruction.get("price_upper")
            if raw_upper not in (None, ""):
                price_upper = float(raw_upper)
                if not math.isfinite(price_upper) or price_upper < 0:
                    raise ValueError(
                        f"invalid precomputed price_upper for ASIN {asin}: "
                        f"{raw_upper!r}"
                    )
            else:
                # Persona rows normally carry an LLM-derived value. Standard
                # rows intentionally store null and use this legacy parser.
                price_upper = explicit_budget_from_instruction(simple_instruction)
                if price_upper is None and simple_instruction != complete_instruction:
                    price_upper = explicit_budget_from_instruction(complete_instruction)
            task_id = int(instruction.get("task_id", len(goals)))
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate task_id in product corpus: {task_id}")
            seen_task_ids.add(task_id)
            goal = {
                "task_id": task_id,
                "task_uid": instruction.get("task_uid"),
                "source_product_index": item.get("source_product_index"),
                "asin": asin,
                "category": item.get("category", ""),
                "name": item.get("title", ""),
                "instruction_text": complete_instruction,
                "instruction_simple": simple_instruction,
                "attributes": attributes,
                "price_upper": price_upper,
                "price_constraint": instruction.get("price_constraint"),
                "goal_options": list(
                    instruction.get("instruction_options") or []
                ),
                "user_persona": persona,
                "reason_key": item.get("reason_key"),
                "weight": 1,
            }
            goals.append(goal)
            for attribute in attributes:
                attribute_counts[attribute] += 1

    print("Goal construction skipped:")
    print(skipped_missing_instructions, skipped_missing_attributes)
    return goals


def _category_parts(value: object) -> list[str]:
    return [
        part.strip().casefold()
        for part in re.split(r"\s*(?:›|>|/|\\)\s*", str(value or ""))
        if part.strip()
    ]


def _title_terms(value: object) -> list[str]:
    text = str(value or "")
    try:
        parsed = _nlp()(text)
    except (ImportError, OSError):
        parsed = None
    if parsed is not None:
        terms = [
            token.text.casefold()
            for token in parsed
            if token.pos_ in ("PNOUN", "NOUN", "PROPN") and token.text.strip()
        ]
        if terms:
            return terms
    # Deterministic fallback for installations where the optional spaCy model
    # is unavailable.  It is deliberately lexical rather than semantic.
    latin = re.findall(r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*", text.casefold())
    han_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text)
    han = [
        run[index : index + 2]
        for run in han_runs
        for index in range(max(1, len(run) - 1))
    ]
    return latin + han


def get_type_reward(purchased_product: dict, goal: dict) -> dict:
    """Implement Appendix C.3's complete category/type score ladder."""
    purchased_path = _category_parts(purchased_product.get("category"))
    goal_path = _category_parts(goal.get("category"))
    category_match = len(set(purchased_path) & set(goal_path)) >= 2

    purchased_terms = _title_terms(
        purchased_product.get("title") or purchased_product.get("Title")
    )
    desired_terms = _title_terms(goal.get("name"))
    title_score = (
        len(set(purchased_terms) & set(desired_terms)) / len(set(desired_terms))
        if desired_terms
        else 0.2
    )

    if category_match or title_score > 0.2:
        r_type = 1.0
    elif title_score == 0:
        r_type = 0.0
    elif title_score < 0.1:
        r_type = 0.1
    else:
        r_type = 0.5
    return {
        "r_type": r_type,
        "category_match": category_match,
        "title_score": title_score,
    }


def get_attribute_reward(purchased_product: dict, goal: dict) -> tuple[float, int]:
    purchased_attributes = purchased_product.get("Attributes") or purchased_product.get(
        "attribute"
    ) or []
    goal_attributes = list(goal.get("attributes") or [])
    if not goal_attributes:
        return 1.0, 0

    searchable_text = " ".join(
        [
            str(purchased_product.get("Title") or purchased_product.get("title") or ""),
            " ".join(map(str, purchased_product.get("BulletPoints") or [])),
            str(
                purchased_product.get("Description")
                or purchased_product.get("full_description")
                or ""
            ),
        ]
    ).casefold()
    matches = 0
    for required in goal_attributes:
        direct = any(
            _fuzzy_ratio(actual, required) > 85
            for actual in purchased_attributes
        )
        if direct or str(required).casefold() in searchable_text:
            matches += 1
    return matches / len(goal_attributes), matches


def _goal_option_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)] if value not in (None, "") else []


def get_option_reward(
    purchased_options: object,
    goal_options: object,
) -> tuple[float, int]:
    purchased_values = _goal_option_values(purchased_options)
    required_values = _goal_option_values(goal_options)
    if not required_values:
        return 1.0, 0
    matches = 0
    for required in required_values:
        if any(_fuzzy_ratio(actual, required) > 85 for actual in purchased_values):
            matches += 1
    return matches / len(required_values), matches


def empty_paper_reward_detail(*, termination_reason: str) -> dict:
    """Return paper metrics for an episode that ended without a purchase."""
    return {
        "reward_version": PAPER_REWARD_VERSION,
        "termination_reason": termination_reason,
        "r_loose": 0.0,
        "r_strict": 0.0,
        "r_success": 0,
        "r_type": 0.0,
        "r_att": 0.0,
        "r_option": 0.0,
        "r_price": 0.0,
        "target_asin_match": False,
        "price_verifiable": False,
    }


def get_reward(
    purchased_product: dict,
    goal: dict,
    price=None,
    options=None,
    *,
    price_resolution: dict | None = None,
    verbose: bool = False,
):
    """Compute the paper's loose, strict and binary-success rewards."""
    selected_options = options if isinstance(options, dict) else {}
    if price_resolution is None:
        if price is not None:
            try:
                explicit_price = float(price)
            except (TypeError, ValueError):
                explicit_price = None
            if explicit_price is not None and (
                not math.isfinite(explicit_price) or explicit_price < 0
            ):
                explicit_price = None
            price_resolution = {
                "status": PASS if explicit_price is not None else "unverifiable",
                "price": explicit_price,
                "method": "explicit_price",
            }
        else:
            price_resolution = resolve_variant_price(
                purchased_product, selected_options
            )

    upper = goal.get("price_upper")
    price_verifiable = price_resolution.get("status") == PASS
    if upper is None:
        r_price = 1.0
    elif price_resolution.get("status") == PASS:
        actual_price = price_resolution.get("price")
        r_price = float(
            actual_price is not None
            and math.isfinite(float(actual_price))
            and float(actual_price) <= float(upper)
        )
    else:
        r_price = 0.0

    type_detail = get_type_reward(purchased_product, goal)
    r_attribute, attribute_matches = get_attribute_reward(purchased_product, goal)
    r_option, option_matches = get_option_reward(
        list(selected_options.values()), goal.get("goal_options")
    )
    attribute_count = len(goal.get("attributes") or [])
    option_count = len(_goal_option_values(goal.get("goal_options")))
    denominator = attribute_count + option_count + 1
    r_loose = type_detail["r_type"] * (
        attribute_matches + option_matches + r_price
    ) / denominator
    r_strict = type_detail["r_type"] * r_attribute * r_option * r_price
    r_success = int(
        type_detail["r_type"] == 1.0
        and r_attribute == 1.0
        and r_option == 1.0
        and r_price == 1.0
    )
    detail = {
        "reward_version": PAPER_REWARD_VERSION,
        "termination_reason": "purchase",
        **type_detail,
        "num_attr_matches": attribute_matches,
        "num_option_matches": option_matches,
        "r_att": r_attribute,
        "r_option": r_option,
        "r_price": r_price,
        "r_loose": r_loose,
        "r_strict": r_strict,
        "r_success": r_success,
        "target_asin_match": str(purchased_product.get("asin"))
        == str(goal.get("asin")),
        "price_verifiable": price_verifiable,
        "price_upper": upper,
        "price_resolution": price_resolution,
    }
    primary_reward = detail[PRIMARY_REWARD]
    return (primary_reward, detail) if verbose else primary_reward
