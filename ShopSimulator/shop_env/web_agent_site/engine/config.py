"""Load and validate the paper-aligned ShopSimulator runtime contract."""

from __future__ import annotations

import json
from pathlib import Path

from web_agent_site.engine.budget import BUDGET_PARSER_VERSION
from web_agent_site.engine.goal import PAPER_REWARD_VERSION, PRIMARY_REWARD
from web_agent_site.engine.observation import OBSERVATION_VERSION
from web_agent_site.engine.search import (
    DEFAULT_FIELD_WEIGHTS,
    OPTION_DOCUMENT_VERSION,
    SEARCH_VERSION,
)
from web_agent_site.engine.variant_price import VARIANT_PRICE_VERSION


ENVIRONMENT_VERSION = "shopsimulator-paper-aligned-v8"
TOOL_VERSION = "shopping-tools-paper-v4"
CATALOG_FILTER_VERSION = "single-price-axis-v1"
SEARCH_TOP_K = 150
SEARCH_PAGE_SIZE = 20
PAPER_REPORTED_METRICS = [
    "r_loose",
    "r_strict",
    "r_success",
    "r_type",
    "r_att",
    "r_option",
    "r_price",
    "target_asin_match",
    "price_verifiable",
]


def load_config(path):
    config_path = Path(path).resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load environment config {config_path}: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("environment config must be an object")
    if config.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("wrong environment_version")
    search = config.get("search")
    if not isinstance(search, dict) or search.get("version") != SEARCH_VERSION:
        raise ValueError("wrong search version")
    if int(search.get("top_k", 0)) != SEARCH_TOP_K:
        raise ValueError(f"search top_k must equal {SEARCH_TOP_K}")
    if int(search.get("page_size", 0)) != SEARCH_PAGE_SIZE:
        raise ValueError(f"search page_size must equal {SEARCH_PAGE_SIZE}")
    if search.get("field_weights") != DEFAULT_FIELD_WEIGHTS:
        raise ValueError("search field weights differ from the index contract")
    if search.get("option_document_version") != OPTION_DOCUMENT_VERSION:
        raise ValueError("wrong search option document version")

    reward = config.get("reward")
    if not isinstance(reward, dict) or reward.get("version") != PAPER_REWARD_VERSION:
        raise ValueError("wrong paper reward version")
    if reward.get("primary") != PRIMARY_REWARD:
        raise ValueError(
            f"the environment's primary reward must be {PRIMARY_REWARD}"
        )
    if reward.get("reported_metrics") != PAPER_REPORTED_METRICS:
        raise ValueError("paper reward reported_metrics differ from the contract")

    limits = config.get("episode_limits")
    if not isinstance(limits, dict):
        raise ValueError("episode_limits must be an object")
    if int(limits.get("single_turn", 0)) != 30:
        raise ValueError("single-turn action limit must equal the paper's 30")
    if int(limits.get("multi_turn", 0)) != 40:
        raise ValueError("multi-turn action limit must equal the paper's 40")

    catalog_filter = config.get("catalog_filter")
    if (
        not isinstance(catalog_filter, dict)
        or catalog_filter.get("version") != CATALOG_FILTER_VERSION
        or int(catalog_filter.get("max_price_affecting_axes", -1)) != 1
    ):
        raise ValueError("wrong catalog price-axis filter contract")

    expected_versions = {
        "budget_parser_version": BUDGET_PARSER_VERSION,
        "variant_price_version": VARIANT_PRICE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "tool_version": TOOL_VERSION,
    }
    for name, expected in expected_versions.items():
        if config.get(name) != expected:
            raise ValueError(f"wrong {name}: expected {expected!r}")
    return config
