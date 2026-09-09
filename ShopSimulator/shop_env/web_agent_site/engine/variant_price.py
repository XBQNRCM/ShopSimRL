"""Deterministic SKU-price resolution for the paper-aligned environment.

The Taobao snapshot stores a listing price range and a price on every option
entry.  A listing range is not a purchasable price, so evaluation resolves the
price of the option selected by the agent.  Products for which two or more
independent option axes affect price are filtered from the catalog because the
released data does not contain a reliable cross-axis price table.
"""

from __future__ import annotations

import math
import re
import unicodedata


VARIANT_PRICE_VERSION = "variant-price-paper-v2"
PASS = "pass"
UNVERIFIABLE = "unverifiable"

def normalize_option_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/", "|")
    return re.sub(r"\s+", "", text)

def _finite_price(value: object) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def option_axes(product: dict) -> dict[str, dict]:
    """Index option values under their exact source axis labels."""
    axes: dict[str, dict] = {}
    raw = product.get("customization_options") or {}
    if not isinstance(raw, dict):
        raw = {}
    for raw_axis, entries in raw.items():
        axis = str(raw_axis)
        values = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            normalized = normalize_option_text(entry.get("value"))
            if normalized:
                values[normalized] = {
                    "value": str(entry.get("value")),
                    "price": _finite_price(entry.get("price")),
                }
        record = {
            "source_axis": axis,
            "values": values,
        }
        axes[axis] = record
    return axes


def price_affecting_axes(product: dict) -> list[str]:
    """Return axes that contain more than one distinct listed price."""
    indexed = option_axes(product)
    result = []
    for source_axis, axis in indexed.items():
        prices = {
            entry["price"]
            for entry in axis["values"].values()
            if entry["price"] is not None
        }
        if len(prices) > 1:
            result.append(source_axis)
    return result


def resolve_variant_price(product: dict, selected_options: object) -> dict:
    """Resolve one evidence-backed purchase price without guessing."""
    if not isinstance(selected_options, dict):
        return {
            "status": UNVERIFIABLE,
            "price": None,
            "version": VARIANT_PRICE_VERSION,
            "method": "invalid_selected_options",
            "evidence": {"reason": "selected_options_not_object"},
        }
    selected = {str(axis): str(value) for axis, value in selected_options.items()}
    indexed = option_axes(product)

    effective_axes = price_affecting_axes(product)
    if len(effective_axes) >= 2:
        # This should be unreachable for a prepared catalog.  Keep the guard
        # so a mismatched dataset cannot silently receive a guessed price.
        return {
            "status": UNVERIFIABLE,
            "price": None,
            "version": VARIANT_PRICE_VERSION,
            "method": "multiple_price_affecting_axes",
            "evidence": {"price_affecting_axes": effective_axes},
        }

    if len(effective_axes) == 1:
        price_axis = effective_axes[0]
        selection = selected.get(price_axis)
        axis = indexed[price_axis]
        if selection is None:
            return {
                "status": UNVERIFIABLE,
                "price": None,
                "version": VARIANT_PRICE_VERSION,
                "method": "price_axis_unselected",
                "evidence": {"price_axis": price_axis},
            }
        entry = axis["values"].get(normalize_option_text(selection))
        price = entry["price"] if entry else None
        if price is None:
            return {
                "status": UNVERIFIABLE,
                "price": None,
                "version": VARIANT_PRICE_VERSION,
                "method": "selected_variant_price_missing",
                "evidence": {"price_axis": price_axis},
            }
        return {
            "status": PASS,
            "price": price,
            "version": VARIANT_PRICE_VERSION,
            "method": "selected_price_axis",
            "evidence": {"price_axis": price_axis},
        }

    option_prices = {
        entry["price"]
        for axis in indexed.values()
        for entry in axis["values"].values()
        if entry["price"] is not None
    }
    if len(option_prices) == 1:
        return {
            "status": PASS,
            "price": next(iter(option_prices)),
            "version": VARIANT_PRICE_VERSION,
            "method": "constant_option_price",
            "evidence": {},
        }

    pricing = [
        price
        for price in (_finite_price(value) for value in product.get("pricing") or [])
        if price is not None
    ]
    if not indexed and len(set(pricing)) == 1:
        return {
            "status": PASS,
            "price": pricing[0],
            "version": VARIANT_PRICE_VERSION,
            "method": "single_listing_price",
            "evidence": {},
        }
    return {
        "status": UNVERIFIABLE,
        "price": None,
        "version": VARIANT_PRICE_VERSION,
        "method": "price_not_deterministic",
        "evidence": {"candidate_prices": sorted(option_prices)},
    }
