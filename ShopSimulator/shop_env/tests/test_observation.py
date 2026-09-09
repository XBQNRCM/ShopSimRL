import unittest
from pathlib import Path

from bs4 import BeautifulSoup

from web_agent_site.engine.engine import ACTION_TO_TEMPLATE
from web_agent_site.engine.observation import (
    StructuredObservationError,
    build_observation_state,
    render_observation_state,
)
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv


ITEM_PAGE = Path(__file__).resolve().parents[1] / "web_agent_site" / "templates" / "item_page.html"


class _SessionServer:
    def __init__(self, options=None):
        self.user_sessions = {"session": {"options": options or {}}}


class ObservationV7Test(unittest.TestCase):
    def test_search_submit_button_is_not_a_click_action(self):
        env = WebAgentTextEnv.__new__(WebAgentTextEnv)
        env.session = "session"
        env.server = _SessionServer()
        env._parse_html = lambda: BeautifulSoup(
            """
            <input id="search_input">
            <button class="btn">Search</button>
            """,
            "html.parser",
        )
        actions = env.get_available_actions()
        self.assertTrue(actions["has_search_bar"])
        self.assertEqual(actions["clickables"], [])

    def test_search_page_preserves_all_twenty_products_and_ranks(self):
        products = {
            f"{index:012d}": {
                "asin": f"{index:012d}",
                "title": f"商品 {index}",
                "category": "测试",
                "pricing": [index],
            }
            for index in range(1, 41)
        }
        state = build_observation_state(
            page_type="search_results",
            session={
                "keywords": ["商品"],
                "normalized_query": "商品",
                "page": 2,
                "total_pages": 2,
                "total_results": 40,
                "search_result_asins": list(products),
                "current_page_asins": list(products)[20:40],
            },
            product_item_dict=products,
            available_actions={
                "has_search_bar": False,
                "clickables": ["< prev", *list(products)[20:40]],
            },
        )
        self.assertEqual(len(state["products"]), 20)
        self.assertEqual(state["search_text"], "商品")
        self.assertEqual(state["normalized_search_text"], "商品")
        self.assertNotIn("query", state)
        self.assertEqual(state["rank_start"], 21)
        self.assertEqual(state["rank_end"], 40)
        self.assertEqual(
            {product["asin"] for product in state["products"]},
            set(state["actions"]) - {"< prev"},
        )

    def test_builder_has_no_goal_parameter_or_hidden_answer(self):
        state = build_observation_state(
            page_type="search_home",
            session={},
            product_item_dict={},
            available_actions={"has_search_bar": True, "clickables": ["search"]},
        )
        self.assertEqual(state["observation_version"], "shopping-observation-v7")
        self.assertTrue(state["search_available"])
        self.assertEqual(state["actions"], [])
        self.assertNotIn("goal", state)
        self.assertNotIn("reward", state)

    def test_empty_webshop_information_pages_are_not_agent_actions(self):
        self.assertEqual(ACTION_TO_TEMPLATE, {"Attributes": "attributes_page.html"})
        template = ITEM_PAGE.read_text(encoding="utf-8")
        for action in ("Description", "Features", "Reviews"):
            self.assertNotIn(f">{action}</button>", template)

    def test_product_state_reports_selected_and_missing_option_axes(self):
        product = {
            "asin": "item",
            "title": "测试商品",
            "options": {
                "颜色分类": ["黑色", "白色"],
                "尺码": ["m", "l"],
            },
        }
        state = build_observation_state(
            page_type="product_detail",
            session={
                "asin": "item",
                "options": {"颜色分类": "黑色"},
                "last_action": {
                    "action": "click[颜色分类=黑色]",
                    "valid": True,
                    "reason": "executed",
                    "message": "点击已执行",
                },
            },
            product_item_dict={"item": product},
            available_actions={
                "has_search_bar": False,
                "clickables": ["颜色分类=白色", "尺码=m", "尺码=l"],
            },
        )
        self.assertEqual(state["selected_options"], {"颜色分类": "黑色"})
        self.assertEqual(state["missing_option_axes"], ["尺码"])
        self.assertFalse(state["options_complete"])
        self.assertEqual(state["last_action"]["reason"], "executed")

    def test_option_actions_include_axis_and_exclude_current_selection(self):
        env = WebAgentTextEnv.__new__(WebAgentTextEnv)
        env.session = "session"
        env.server = _SessionServer({"颜色分类": "共同值"})
        env._parse_html = lambda: BeautifulSoup(
            """
            <input type="radio" name="颜色分类" value="共同值" checked>
            <input type="radio" name="颜色分类" value="白色">
            <input type="radio" name="尺码" value="共同值">
            """,
            "html.parser",
        )
        actions = env.get_available_actions()["clickables"]
        self.assertNotIn("颜色分类=共同值", actions)
        self.assertIn("颜色分类=白色", actions)
        self.assertIn("尺码=共同值", actions)

    def test_item_template_gates_purchase_on_complete_options(self):
        template = ITEM_PAGE.read_text(encoding="utf-8")
        self.assertIn("{% if options_complete %}", template)
        self.assertIn("尚未选择规格轴", template)

    def test_search_home_renderer_exposes_search_without_fake_click(self):
        state = build_observation_state(
            page_type="search_home",
            session={},
            product_item_dict={},
            available_actions={"has_search_bar": True, "clickables": []},
        )
        observation = render_observation_state(state)
        self.assertTrue(observation.startswith("页面：搜索首页"))
        self.assertIn("页面：搜索首页", observation)
        self.assertIn("页面能力：\n- 支持商品搜索", observation)
        self.assertNotIn("search[", observation)
        self.assertNotIn("click[", observation)
        self.assertNotIn("[SHOPPING_OBSERVATION", observation)
        self.assertNotIn("[SEP]", observation)

    def test_search_results_renderer_preserves_hierarchy_and_exact_actions(self):
        asin = "000000000001"
        state = build_observation_state(
            page_type="search_results",
            session={
                "keywords": ["降噪", "耳机"],
                "normalized_query": "降噪耳机",
                "page": 1,
                "total_pages": 1,
                "total_results": 1,
                "search_result_asins": [asin],
                "current_page_asins": [asin],
            },
            product_item_dict={
                asin: {
                    "asin": asin,
                    "title": "测试降噪耳机",
                    "brand": "测试品牌",
                    "category": "耳机",
                    "pricing": [299],
                    "attribute": ["主动降噪", "蓝牙"],
                }
            },
            available_actions={"has_search_bar": False, "clickables": [asin]},
        )
        observation = render_observation_state(state)
        self.assertIn("搜索词：降噪 耳机", observation)
        self.assertIn("1. 测试降噪耳机", observation)
        self.assertIn(f"ASIN：{asin}", observation)
        self.assertIn(f"可点击元素：\n- {asin}", observation)
        self.assertNotIn(f"click[{asin}]", observation)
        self.assertNotIn("normalized_search_text", observation)

    def test_product_renderer_reports_selection_and_invalid_action(self):
        asin = "000000000002"
        state = build_observation_state(
            page_type="product_detail",
            session={
                "asin": asin,
                "options": {"颜色分类": "黑色"},
                "last_action": {
                    "action": "click[buy now]",
                    "valid": False,
                    "reason": "incomplete_options",
                    "message": "购买前还需选择规格轴：尺码",
                },
            },
            product_item_dict={
                asin: {
                    "asin": asin,
                    "title": "测试商品",
                    "options": {
                        "颜色分类": ["黑色", "白色"],
                        "尺码": ["M", "L"],
                    },
                }
            },
            available_actions={
                "has_search_bar": False,
                "clickables": ["颜色分类=白色", "尺码=M", "尺码=L"],
            },
        )
        observation = render_observation_state(state)
        self.assertIn("颜色分类：已选择 黑色", observation)
        self.assertIn("尺码：尚未选择", observation)
        self.assertIn("仍需选择：尺码", observation)
        self.assertIn("上次交互结果：", observation)
        self.assertIn("结果：无效", observation)
        self.assertIn("可点击元素：", observation)
        self.assertIn("- 尺码=M", observation)
        self.assertNotIn("click[", observation)

    def test_renderer_rejects_hidden_answer_fields(self):
        with self.assertRaises(StructuredObservationError):
            render_observation_state(
                {
                    "observation_version": "shopping-observation-v7",
                    "page_type": "search_home",
                    "search_available": True,
                    "actions": [],
                    "reward": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
