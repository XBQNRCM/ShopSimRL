#!/usr/bin/env python3
"""Run a no-LLM end-to-end reset/search/click/purchase smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))
os.environ.setdefault(
    "SHOP_ENV_CONFIG", str(SHOP_ENV / "configs" / "environment.json")
)
os.environ.setdefault(
    "SHOP_SEARCH_INDEX", str(SHOP_ENV / "search_engine" / "products.sqlite3")
)

from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402
from web_agent_site.engine.options import format_option_action  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = WebAgentTextEnv(
        observation_mode="structured_text",
        split="train",
        session_prefix="smoke",
    )
    env.reset(idx=args.task, if_persona=False)
    goal = env.server.goals_by_task_id[args.task]

    _, status, _ = env.step(f"search[{goal['name']}]")
    if status["done"]:
        raise RuntimeError(f"search terminated unexpectedly: {status}")

    target = str(goal["asin"]).lower()
    for _ in range(20):
        actions = env.get_available_actions()["clickables"]
        if target in actions:
            break
        if "next >" not in actions:
            raise RuntimeError("gold product was not present in reproducible search results")
        env.step("click[next >]")
    else:
        raise RuntimeError("gold product search exceeded the page safety limit")

    env.step(f"click[{target}]")
    product_url = env.browser.current_url
    _, invalid_search, _ = env.step("search[当前页不应允许搜索]")
    if invalid_search["action_feedback"]["reason"] != "search_unavailable":
        raise RuntimeError(f"product-page search was not rejected: {invalid_search}")
    if env.browser.current_url != product_url:
        raise RuntimeError("invalid search changed the product page")

    _, premature_buy, _ = env.step("click[buy now]")
    if premature_buy["action_feedback"]["reason"] != "incomplete_options":
        raise RuntimeError(f"incomplete purchase was not rejected: {premature_buy}")

    desired_values = {
        str(option).strip().replace("/", " | ").casefold()
        for option in goal.get("goal_options") or ()
    }
    state = env.structured_observation()
    for axis, values in state.get("available_options", {}).items():
        selected = next(
            (value for value in values if value.casefold() in desired_values),
            values[0],
        )
        action = format_option_action(axis, selected)
        _, option_status, _ = env.step(f"click[{action}]")
        if not option_status["action_feedback"]["valid"]:
            raise RuntimeError(f"option selection failed: {option_status}")
        _, repeated_status, _ = env.step(f"click[{action}]")
        if repeated_status["action_feedback"]["reason"] != "already_selected":
            raise RuntimeError(f"repeated option was not rejected: {repeated_status}")

    _, status, _ = env.step("click[buy now]")
    if not status["done"]:
        raise RuntimeError(
            "purchase did not terminate the environment: "
            f"status={status}, state={env.structured_observation()}"
        )
    detail = status.get("reward_detail") or {}
    if detail.get("reward_version") != "shopsimulator-paper-reward-v2":
        raise RuntimeError(f"unexpected reward contract: {status}")
    for metric in ("r_loose", "r_strict", "r_success", "r_type", "r_att", "r_option", "r_price"):
        if metric not in detail:
            raise RuntimeError(f"missing paper metric {metric}: {status}")
    if status["reward"] != detail["r_strict"]:
        raise RuntimeError(f"primary reward is not r_strict: {status}")
    print(
        json.dumps(
            {
                "task": args.task,
                "asin": target,
                "reward": status["reward"],
                "termination_reason": status.get("termination_reason"),
                "environment_version": env.server.environment_version,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
