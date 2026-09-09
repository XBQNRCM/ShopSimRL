"""Public option-selection rules shared by rendering and Agent actions."""

from __future__ import annotations


OPTION_ACTION_SEPARATOR = "="


def product_option_axes(product: dict) -> dict[str, list[str]]:
    """Return the product's public axes without renaming their source labels."""
    rendered = product.get("options")
    if isinstance(rendered, dict):
        return {
            str(axis): [str(value) for value in values or []]
            for axis, values in rendered.items()
        }

    raw = product.get("customization_options") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(axis): [
            str(entry.get("value"))
            for entry in entries or []
            if isinstance(entry, dict) and entry.get("value") not in (None, "")
        ]
        for axis, entries in raw.items()
    }


def format_option_action(axis: object, value: object) -> str:
    """Disambiguate identical values that occur under different axes."""
    return f"{axis}{OPTION_ACTION_SEPARATOR}{value}"


def option_selection_state(product: dict, selected_options: object) -> dict:
    """Describe selected, missing, and invalid axes using exact source labels."""
    axes = product_option_axes(product)
    selected = selected_options if isinstance(selected_options, dict) else {}
    valid_selected = {}
    invalid_selected = []

    for raw_axis, raw_value in selected.items():
        axis = str(raw_axis)
        value = str(raw_value)
        if axis not in axes:
            invalid_selected.append(
                {"axis": axis, "value": value, "reason": "unknown_axis"}
            )
        elif value not in axes[axis]:
            invalid_selected.append(
                {"axis": axis, "value": value, "reason": "unknown_value"}
            )
        else:
            valid_selected[axis] = value

    missing_axes = [axis for axis in axes if axis not in valid_selected]
    return {
        "selected_options": valid_selected,
        "missing_option_axes": missing_axes,
        "invalid_selected_options": invalid_selected,
        "options_complete": not missing_axes and not invalid_selected,
    }
