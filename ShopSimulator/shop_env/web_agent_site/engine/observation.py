"""Build and render the canonical, answer-free ShopSimulator observation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from web_agent_site.engine.options import option_selection_state
from web_agent_site.engine.product_id import is_product_id


OBSERVATION_VERSION = "shopping-observation-v7"


class StructuredObservationError(ValueError):
    """The environment supplied malformed or unsafe public state."""


def _compact_list(value, limit=10):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item) for item in value[:limit] if str(item).strip()]


def _text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _public_list(value):
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _public_mapping(value, field_name):
    if not isinstance(value, Mapping):
        raise StructuredObservationError(f"{field_name} must be an object")
    return value


def _display_value(value):
    if value is None or value == "":
        return "未提供"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _text(value)


def _render_last_action(state):
    feedback = state.get("last_action")
    if not isinstance(feedback, Mapping):
        return []
    valid = bool(feedback.get("valid"))
    lines = [
        "上次交互结果：",
        f"- 结果：{'成功' if valid else '无效'}",
    ]
    message = _text(feedback.get("message"))
    reason = _text(feedback.get("reason"))
    if message:
        lines.append(f"- 说明：{message}")
    if reason and reason != "executed":
        lines.append(f"- 原因代码：{reason}")
    return lines


def _render_interactions(state):
    actions = _public_list(state.get("actions"))
    lines = []
    if bool(state.get("search_available")):
        lines.extend(["页面能力：", "- 支持商品搜索"])
    if actions:
        if lines:
            lines.append("")
        lines.append("可点击元素：")
        lines.extend(f"- {action}" for action in actions)
    if not lines:
        lines.append("当前页面没有可交互元素。")
    return lines


def _render_search_results(state):
    products = state.get("products")
    if not isinstance(products, list):
        raise StructuredObservationError("search results must contain a products list")

    actions = set(_public_list(state.get("actions")))
    product_asins = []
    lines = [
        "页面：搜索结果",
        f"搜索词：{_text(state.get('search_text')) or '未提供'}",
        (
            f"结果：第 {int(state.get('page', 1))}/{int(state.get('total_pages', 1))} 页，"
            f"共 {int(state.get('total_results', 0))} 项，"
            f"当前显示排名 {int(state.get('rank_start', 0))}-"
            f"{int(state.get('rank_end', 0))}"
        ),
        "",
        "商品列表：",
    ]
    for product in products:
        product = _public_mapping(product, "each search product")
        asin = _text(product.get("asin"))
        if not is_product_id(asin):
            raise StructuredObservationError(
                f"invalid search-result ASIN: {asin!r}"
            )
        product_asins.append(asin)
        rank = int(product.get("rank", 0))
        lines.extend(
            [
                f"{rank}. {_text(product.get('title')) or '未命名商品'}",
                f"   ASIN：{asin}",
                f"   价格：{_display_value(product.get('price'))}",
                f"   品牌：{_text(product.get('brand')) or '未提供'}",
                f"   类别：{_text(product.get('category')) or '未提供'}",
            ]
        )
        attributes = _public_list(product.get("key_attributes"))
        if attributes:
            lines.append("   关键属性：" + "；".join(attributes))

    actionable_asins = {action for action in actions if is_product_id(action)}
    if set(product_asins) != actionable_asins:
        raise StructuredObservationError(
            "model-visible search ASINs differ from environment-actionable ASINs"
        )
    if len(product_asins) > 20:
        raise StructuredObservationError(
            "search page exceeds the frozen page size of 20"
        )
    if not product_asins:
        lines.append("（本页没有商品）")
    return lines


def _render_product(state):
    product = _public_mapping(state.get("product"), "product")
    asin = _text(product.get("asin"))
    if not is_product_id(asin):
        raise StructuredObservationError(f"invalid product ASIN: {asin!r}")

    selected = _public_mapping(
        state.get("selected_options") or {}, "selected_options"
    )
    available = _public_mapping(
        state.get("available_options") or {}, "available_options"
    )
    missing_axes = _public_list(state.get("missing_option_axes"))
    page_name = (
        "商品属性页"
        if state.get("page_type") == "information_subpage"
        else "商品详情"
    )
    lines = [
        f"页面：{page_name}",
        "商品：",
        f"- 标题：{_text(product.get('title')) or '未命名商品'}",
        f"- ASIN：{asin}",
        f"- 价格：{_display_value(state.get('selected_price', product.get('price')))}",
        f"- 品牌：{_text(product.get('brand')) or '未提供'}",
        f"- 类别：{_text(product.get('category')) or '未提供'}",
    ]
    attributes = _public_list(product.get("key_attributes"))
    if attributes:
        lines.append("- 关键属性：" + "；".join(attributes))

    lines.extend(["", "规格选择："])
    if available:
        for axis, values in available.items():
            axis_text = _text(axis)
            selected_value = _text(selected.get(axis))
            status = f"已选择 {selected_value}" if selected_value else "尚未选择"
            lines.append(f"- {axis_text}：{status}")
            public_values = _public_list(values)
            if public_values:
                lines.append("  可选值：" + " / ".join(public_values))
    else:
        lines.append("- 此商品没有规格选项")

    if missing_axes:
        lines.append("仍需选择：" + "、".join(missing_axes))
    elif bool(state.get("options_complete")):
        lines.append("规格状态：已选择全部规格，可以购买")

    invalid_selected = state.get("invalid_selected_options")
    if isinstance(invalid_selected, Mapping) and invalid_selected:
        lines.append(
            "无效的历史规格选择："
            + json.dumps(invalid_selected, ensure_ascii=False, sort_keys=True)
        )

    if state.get("page_type") == "information_subpage":
        lines.extend(
            [
                "",
                f"子页面：{_text(state.get('subpage')) or 'attributes'}",
                "内容：" + _display_value(state.get("content")),
            ]
        )
    return lines


def render_observation_state(state: Mapping) -> str:
    """Render the public state into the sole model-facing page observation."""
    if not isinstance(state, Mapping):
        raise StructuredObservationError("observation_state must be an object")
    if state.get("observation_version") != OBSERVATION_VERSION:
        raise StructuredObservationError("unsupported observation_state version")
    forbidden = {"goal", "reward", "reward_detail", "target_asin", "answer"}
    leaked = forbidden.intersection(state)
    if leaked:
        raise StructuredObservationError(
            "observation_state contains forbidden fields: "
            + ", ".join(sorted(leaked))
        )

    page_type = str(state.get("page_type") or "unknown")
    if page_type == "search_home":
        lines = ["页面：搜索首页"]
    elif page_type == "search_results":
        lines = _render_search_results(state)
    elif page_type in {"product_detail", "information_subpage"}:
        lines = _render_product(state)
    elif page_type == "terminal":
        lines = ["页面：任务已终止", "不能再执行操作。"]
    else:
        raise StructuredObservationError(f"unsupported page_type: {page_type!r}")

    last_action_lines = _render_last_action(state)
    if last_action_lines:
        lines.extend(["", *last_action_lines])
    lines.extend(["", *_render_interactions(state)])
    return "\n".join(lines).strip() + "\n"


def product_summary(product: dict, *, rank: int | None = None) -> dict:
    pricing = product.get("Price")
    if pricing is None:
        pricing = product.get("pricing")
    result = {
        "asin": str(product.get("asin", "")),
        "title": str(product.get("title") or product.get("Title") or ""),
        "brand": str(product.get("brand") or product.get("shop_name") or ""),
        "category": str(product.get("category") or ""),
        "price": pricing,
        "key_attributes": _compact_list(
            product.get("attribute") or product.get("Attributes") or []
        ),
    }
    if rank is not None:
        result["rank"] = int(rank)
    return result


def build_observation_state(
    *,
    page_type: str,
    session: dict,
    product_item_dict: dict,
    available_actions: dict,
) -> dict:
    """Build one public state; the task goal is intentionally not accepted."""
    clickables = [
        str(value)
        for value in available_actions.get("clickables", [])
        if str(value).casefold() != "search"
    ]
    state = {
        "observation_version": OBSERVATION_VERSION,
        "page_type": page_type,
        "search_available": bool(available_actions.get("has_search_bar")),
        "actions": clickables,
    }
    last_action = session.get("last_action")
    if isinstance(last_action, dict):
        state["last_action"] = dict(last_action)
    if page_type == "search_results":
        all_asins = session.get("search_result_asins") or []
        page_asins = session.get("current_page_asins") or []
        rank_by_asin = {
            asin: index for index, asin in enumerate(all_asins, start=1)
        }
        state.update(
            {
                "search_text": " ".join(session.get("keywords") or []),
                "normalized_search_text": session.get("normalized_query") or "",
                "page": int(session.get("page") or 1),
                "total_pages": int(session.get("total_pages") or 1),
                "total_results": int(session.get("total_results") or 0),
                "rank_start": min(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "rank_end": max(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "products": [
                    product_summary(
                        product_item_dict[asin],
                        rank=rank_by_asin[asin],
                    )
                    for asin in page_asins
                    if asin in product_item_dict and asin in rank_by_asin
                ],
            }
        )
    elif page_type in {"product_detail", "information_subpage"}:
        asin = session.get("asin")
        product = product_item_dict.get(asin, {})
        state["product"] = product_summary(product)
        selection = option_selection_state(product, session.get("options"))
        state["selected_options"] = selection["selected_options"]
        state["available_options"] = {
            str(key): _compact_list(values, limit=100)
            for key, values in (product.get("options") or {}).items()
        }
        state["missing_option_axes"] = selection["missing_option_axes"]
        state["options_complete"] = selection["options_complete"]
        if selection["invalid_selected_options"]:
            state["invalid_selected_options"] = selection[
                "invalid_selected_options"
            ]
        if session.get("selected_price") is not None:
            state["selected_price"] = session["selected_price"]
        if page_type == "information_subpage":
            subpage = str(session.get("subpage") or "information")
            field = {"attributes": "Attributes"}.get(subpage.casefold())
            state["subpage"] = subpage
            state["content"] = product.get(field, "") if field else ""
    return state


def page_type_from_name(page_name: str) -> str:
    return {
        "": "search_home",
        "search_results": "search_results",
        "item_page": "product_detail",
        "item_sub_page": "information_subpage",
        "done": "terminal",
    }.get(page_name, "unknown")
