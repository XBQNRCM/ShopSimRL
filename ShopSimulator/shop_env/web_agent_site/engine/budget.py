"""Extract an explicit price ceiling from a Chinese shopping instruction."""

from __future__ import annotations

import re


BUDGET_PARSER_VERSION = "instruction-budget-v1"

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_LARGE_UNITS = {"万": 10000, "亿": 100000000}
_CN_CHARS = "零〇一二两三四五六七八九十百千万亿"
_NUMBER = rf"(?:\d+(?:\.\d+)?|[{_CN_CHARS}]+)"
_SCALE = r"(?:万|千|[kK])?"
_CURRENCY = r"(?:元|块钱|块)"
_PRICE_PREFIX = r"(?:预算|价格|价钱|价位|售价|单价|花费|费用|总价|预期价格)"
_APPROX = {"左右", "上下", "大约", "上下浮动", "出头"}


def _chinese_number(text: str) -> float:
    if all(character in _CN_DIGITS for character in text):
        return float("".join(str(_CN_DIGITS[character]) for character in text))

    shorthand = re.fullmatch(
        rf"([{''.join(_CN_DIGITS)}]+)([十百千万])([{''.join(_CN_DIGITS)}])",
        text,
    )
    if shorthand:
        unit = {**_CN_SMALL_UNITS, **_CN_LARGE_UNITS}[shorthand.group(2)]
        return (
            _chinese_number(shorthand.group(1)) * unit
            + _chinese_number(shorthand.group(3)) * unit / 10
        )

    total = 0
    section = 0
    number = 0
    for character in text:
        if character in _CN_DIGITS:
            number = _CN_DIGITS[character]
        elif character in _CN_SMALL_UNITS:
            unit = _CN_SMALL_UNITS[character]
            section += (number or 1) * unit
            number = 0
        elif character in _CN_LARGE_UNITS:
            unit = _CN_LARGE_UNITS[character]
            section += number
            total = (total + section) * unit
            section = number = 0
    return float(total + section + number)


def _scaled(number: str, unit: str | None) -> float:
    value = (
        float(number)
        if re.fullmatch(r"\d+(?:\.\d+)?", number)
        else _chinese_number(number)
    )
    normalized = str(unit or "").casefold()
    if normalized == "万":
        value *= 10000
    elif normalized in {"千", "k"}:
        value *= 1000
    return value


def _upper_with_tolerance(value: float, qualifier: str | None) -> float:
    return value * 1.1 if qualifier in _APPROX else value


def explicit_budget_from_instruction(instruction: object) -> float | None:
    """Return a user-stated upper price, never one derived from the Gold item.

    Ranges use their upper endpoint. Approximate targets such as ``70元左右``
    use a frozen +10% tolerance. A lower bound such as ``4k+`` or ``超过20元``
    is not an upper budget and therefore returns ``None``.
    """
    text = re.sub(r"[,，]", "", str(instruction or ""))

    # Colloquial shorthand: 预算1万2以内 -> 12000.
    shorthand = re.search(
        rf"{_PRICE_PREFIX}(?:金额)?(?:控制)?(?:在|为|是)?\s*"
        rf"(\d+)\s*万\s*(\d+)\s*(?:千)?\s*"
        rf"(以内|以下|之内|左右|上下)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        return _upper_with_tolerance(value, shorthand.group(3))

    unqualified_range = re.search(
        rf"{_PRICE_PREFIX}(?:金额)?"
        rf"(?:控制)?(?:在|为|是|要在|大概|约|差不多)?\s*"
        rf"({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})?\s*"
        rf"(?:-|~|～|—|至|到|和)\s*"
        rf"({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})?\s*"
        rf"(?:之间|以内|以下|之内)",
        text,
    )
    if unqualified_range:
        low = _scaled(unqualified_range.group(1), unqualified_range.group(2))
        high = _scaled(unqualified_range.group(3), unqualified_range.group(4))
        if low > 0 and high >= low:
            return high

    # A range may write the currency once or at both endpoints.
    price_range = re.search(
        rf"(?:{_PRICE_PREFIX})?(?:金额)?"
        rf"(?:控制)?(?:在|为|是|要在|大概|约|差不多)?\s*"
        rf"({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})?\s*"
        rf"(?:-|~|～|—|至|到|和)\s*"
        rf"({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})\s*"
        rf"(?:之间|以内|以下|之内|左右|上下)?",
        text,
    )
    if price_range:
        low = _scaled(price_range.group(1), price_range.group(2))
        high = _scaled(price_range.group(3), price_range.group(4))
        if low > 0 and high >= low:
            return high

    upper_bound = re.search(
        rf"(?:{_PRICE_PREFIX})?(?:金额)?[^，。；;]{{0,8}}?"
        rf"(?:不得超过|不能超过|不要超过|不能高于|不要高于|不超过|不高于|不高过|别超过|别超|不超|低于|小于|不到|最高(?:为|是)?|上限(?:为|是)?|只能给到)\s*"
        rf"({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})",
        text,
    )
    if upper_bound:
        value = _scaled(upper_bound.group(1), upper_bound.group(2))
        return value if value > 0 else None

    # Explicit lower bounds cannot be converted to the paper's upper limit.
    if re.search(
        rf"{_PRICE_PREFIX}[^，。；;]{{0,8}}?"
        rf"(?:不低于|至少|(?<!不要)(?<!不能)(?<!不)(?<!别)(?:超过|高于))\s*"
        rf"{_NUMBER}\s*{_SCALE}\s*(?:{_CURRENCY})?",
        text,
    ) or re.search(
        rf"{_PRICE_PREFIX}[^，。；;]{{0,8}}?{_NUMBER}\s*{_SCALE}\s*\+",
        text,
    ):
        return None

    prefixed = re.search(
        rf"{_PRICE_PREFIX}(?:金额)?"
        rf"[^\d{_CN_CHARS}，。；;]{{0,12}}?"
        rf"({_NUMBER})\s*({_SCALE})\s*(来|多)?\s*(?:{_CURRENCY})?\s*"
        rf"(以内|以下|之内|左右|上下|大约|上下浮动|内)?",
        text,
    )
    if prefixed:
        value = _scaled(prefixed.group(1), prefixed.group(2))
        qualifier = prefixed.group(4) or ("左右" if prefixed.group(3) else None)
        if value > 0:
            return _upper_with_tolerance(value, qualifier)

    approximate_before = re.search(
        rf"(?:差不多|大约|大概|约)\s*({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})",
        text,
    )
    if approximate_before:
        value = _scaled(approximate_before.group(1), approximate_before.group(2))
        return _upper_with_tolerance(value, "左右") if value > 0 else None

    approximate_middle = re.search(
        rf"({_NUMBER})\s*({_SCALE})\s*(?:左右|上下|大约)\s*(?:{_CURRENCY})",
        text,
    )
    if approximate_middle:
        value = _scaled(approximate_middle.group(1), approximate_middle.group(2))
        return _upper_with_tolerance(value, "左右") if value > 0 else None

    approximate_over = re.search(
        rf"({_NUMBER})\s*({_SCALE})\s*多\s*(?:{_CURRENCY})",
        text,
    )
    if approximate_over:
        value = _scaled(approximate_over.group(1), approximate_over.group(2))
        return _upper_with_tolerance(value, "左右") if value > 0 else None

    large_approximate = re.search(
        rf"({_NUMBER})\s*(左右|上下)\s*(?:的)?\s*(?:预算|价位|价格)",
        text,
    )
    if large_approximate:
        value = _scaled(large_approximate.group(1), None)
        if value >= 100:
            return _upper_with_tolerance(value, large_approximate.group(2))

    implicit_large_approximate = re.search(
        rf"([{_CN_CHARS}]*[千万][{_CN_CHARS}]*)\s*(左右|上下)", text
    )
    if implicit_large_approximate:
        value = _scaled(implicit_large_approximate.group(1), None)
        if value >= 1000:
            return _upper_with_tolerance(
                value, implicit_large_approximate.group(2)
            )

    implicit_affordable = re.search(
        rf"({_NUMBER})\s*(?:以内|以下|内)\s*(?:能)?(?:拿下|搞定)", text
    )
    if implicit_affordable:
        value = _scaled(implicit_affordable.group(1), None)
        if value > 0:
            return value

    implicit_approximate_affordable = re.search(
        rf"({_NUMBER})\s*(?:左右|上下)\s*(?:能)?(?:拿下|搞定)", text
    )
    if implicit_approximate_affordable:
        value = _scaled(implicit_approximate_affordable.group(1), None)
        if value > 0:
            return _upper_with_tolerance(value, "左右")

    prepared_budget = re.search(
        rf"(?:准备(?:了)?|给到)\s*({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})",
        text,
    )
    if prepared_budget:
        value = _scaled(prepared_budget.group(1), prepared_budget.group(2))
        if value > 0:
            return value

    # Currency plus an upper/approximate qualifier is unambiguously a price
    # even when the sentence omits words such as “预算” or “价格”.
    standalone = re.search(
        rf"({_NUMBER})\s*({_SCALE})\s*(?:{_CURRENCY})\s*"
        rf"(以内|以下|之内|左右|上下|大约|上下浮动|内|出头|都不到|拿下|能拿下|帮我搞定|能搞定|搞定|能买|来买|应该够|基本能买到|这个价|的价格|的价位|价位|的预算|预算)",
        text,
    )
    if standalone:
        value = _scaled(standalone.group(1), standalone.group(2))
        return _upper_with_tolerance(value, standalone.group(3)) if value > 0 else None
    vague_teens = re.search(r"([二三四五六七八九]?十)几\s*(?:元|块钱|块)", text)
    if vague_teens:
        base = _chinese_number(vague_teens.group(1))
        return base + 9
    if "个位数价格" in text:
        return 9.0
    return None
