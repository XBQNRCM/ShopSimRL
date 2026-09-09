"""Product-ID validation shared by ShopSimulator runtime components."""

from __future__ import annotations

import re


PRODUCT_ID_PATTERN = re.compile(r"^\d{8,12}$")


def is_product_id(value: object) -> bool:
    return PRODUCT_ID_PATTERN.fullmatch(str(value or "").strip()) is not None
