import unittest
from unittest.mock import patch

from web_agent_site.engine.goal import get_option_reward, get_reward, get_type_reward
from web_agent_site.engine.variant_price import (
    PASS,
    UNVERIFIABLE,
    price_affecting_axes,
    resolve_variant_price,
)


def product(asin="gold", *, price=1999.0):
    return {
        "asin": asin,
        "title": "智能热洗白色洗地机",
        "category": "家电›清洁电器›洗地机",
        "attribute": ["智能", "热洗"],
        "pricing": [price],
        "customization_options": {
            "颜色分类": [
                {"value": "白色", "price": price},
                {"value": "黑色", "price": price + 100},
            ],
            "尺码": [
                {"value": "标准版", "price": price},
                {"value": "大号", "price": price},
            ],
        },
    }


def goal():
    return {
        "asin": "gold",
        "name": "智能热洗白色洗地机",
        "category": "家电›清洁电器›洗地机",
        "attributes": ["智能", "热洗"],
        "goal_options": ["白色"],
        "price_upper": 2100.0,
    }


class PaperRewardTest(unittest.TestCase):
    def test_loose_strict_and_success_follow_paper_formula(self):
        reward, detail = get_reward(
            product(),
            goal(),
            options={"颜色分类": "白色"},
            verbose=True,
        )
        self.assertEqual(detail["r_type"], 1.0)
        self.assertEqual(detail["r_att"], 1.0)
        self.assertEqual(detail["r_option"], 1.0)
        self.assertEqual(detail["r_price"], 1.0)
        self.assertEqual(detail["r_loose"], 1.0)
        self.assertEqual(detail["r_strict"], 1.0)
        self.assertEqual(detail["r_success"], 1)
        self.assertEqual(reward, detail["r_strict"])
        self.assertNotIn("query_match", detail)
        self.assertNotIn("reward_type", detail)
        self.assertNotIn("weighted_score", detail)

    def test_target_asin_is_diagnostic_not_a_success_gate(self):
        _, detail = get_reward(
            product("alternative"),
            goal(),
            options={"颜色分类": "白色"},
            verbose=True,
        )
        self.assertFalse(detail["target_asin_match"])
        self.assertEqual(detail["r_success"], 1)

    def test_gold_asin_does_not_override_wrong_option(self):
        reward, detail = get_reward(
            product(),
            goal(),
            options={"颜色分类": "黑色"},
            verbose=True,
        )
        self.assertTrue(detail["target_asin_match"])
        self.assertEqual(detail["r_option"], 0.0)
        self.assertEqual(detail["r_success"], 0)
        self.assertEqual(detail["r_strict"], 0.0)
        self.assertAlmostEqual(detail["r_loose"], 0.75)
        self.assertEqual(reward, 0.0)

    def test_non_verbose_reward_is_also_strict(self):
        reward = get_reward(
            product(),
            goal(),
            options={"颜色分类": "黑色"},
        )
        self.assertEqual(reward, 0.0)

    def test_type_reward_has_complete_one_half_tenth_zero_ladder(self):
        task_goal = {"category": "甲›乙", "name": "desired"}
        term_map = {
            "desired": [f"t{i}" for i in range(20)],
            "one": ["t0"],
            "two": ["t0", "t1"],
            "none": [],
        }

        def fake_terms(value):
            return term_map[str(value)]

        with patch("web_agent_site.engine.goal._title_terms", side_effect=fake_terms):
            tenth = get_type_reward(
                {"category": "丙›丁", "title": "one"}, task_goal
            )
            half = get_type_reward(
                {"category": "丙›丁", "title": "two"}, task_goal
            )
            zero = get_type_reward(
                {"category": "丙›丁", "title": "none"}, task_goal
            )
            full = get_type_reward(
                {"category": "甲›乙", "title": "none"}, task_goal
            )
        self.assertEqual(tenth["r_type"], 0.1)
        self.assertEqual(half["r_type"], 0.5)
        self.assertEqual(zero["r_type"], 0.0)
        self.assertEqual(full["r_type"], 1.0)

    def test_empty_desired_title_terms_preserve_original_neutral_score(self):
        with patch("web_agent_site.engine.goal._title_terms", return_value=[]):
            detail = get_type_reward(
                {"category": "丙›丁", "title": "none"},
                {"category": "甲›乙", "name": "desired"},
            )
        self.assertEqual(detail["title_score"], 0.2)
        self.assertEqual(detail["r_type"], 0.5)

    def test_option_matching_does_not_apply_webshop_color_normalization(self):
        score, matches = get_option_reward(["深海blue色"], ["天空blue款"])
        self.assertEqual(score, 0.0)
        self.assertEqual(matches, 0)

    def test_price_is_resolved_from_selected_single_price_axis(self):
        candidate = product()
        self.assertEqual(price_affecting_axes(candidate), ["颜色分类"])
        selected = resolve_variant_price(candidate, {"颜色分类": "白色"})
        missing = resolve_variant_price(candidate, {})
        self.assertEqual(selected["status"], PASS)
        self.assertEqual(selected["price"], 1999.0)
        self.assertEqual(missing["status"], UNVERIFIABLE)

    def test_price_axis_requires_the_exact_source_axis_name(self):
        candidate = product()
        aliased = resolve_variant_price(candidate, {"color": "白色"})
        self.assertEqual(aliased["status"], UNVERIFIABLE)
        self.assertEqual(aliased["method"], "price_axis_unselected")
        self.assertEqual(aliased["evidence"]["price_axis"], "颜色分类")

    def test_multiple_price_axes_are_detected_for_catalog_filtering(self):
        candidate = product()
        candidate["customization_options"]["尺码"][1]["price"] = 2299.0
        self.assertEqual(
            set(price_affecting_axes(candidate)), {"颜色分类", "尺码"}
        )
        self.assertEqual(resolve_variant_price(candidate, {})["status"], UNVERIFIABLE)

    def test_budget_failure_and_unverifiable_price_receive_zero_price_score(self):
        over = get_reward(
            product(price=2200),
            goal(),
            options={"颜色分类": "白色"},
            verbose=True,
        )[1]
        unknown = get_reward(
            product(),
            goal(),
            options={},
            verbose=True,
        )[1]
        self.assertEqual(over["r_price"], 0.0)
        self.assertEqual(unknown["r_price"], 0.0)
        self.assertFalse(unknown["price_verifiable"])

    def test_no_budget_does_not_require_or_invent_a_price(self):
        no_budget = goal()
        no_budget["price_upper"] = None
        detail = get_reward(
            product(), no_budget, options={}, verbose=True
        )[1]
        self.assertEqual(detail["r_price"], 1.0)
        self.assertFalse(detail["price_verifiable"])


if __name__ == "__main__":
    unittest.main()
