import unittest

from shop_env.shop_agent import _extract_action_from_response, _handle_reset_action
from web_agent_site.engine.engine import ActionParseError, parse_action


class _DummyServer:
    environment_version = "test-environment"
    goals = [{
        "instruction_text": "完整隐藏目标",
        "goal_options": ["黑色"],
        "reason_key": "hidden",
    }]


class _DummyEnv:
    server = _DummyServer()
    instruction_text = "简化目标"
    instruction_simple = "简化目标"
    if_persona = True
    user_persona = {"偏好": "简洁", "__reasoning__": "hidden chain"}

    def reset(self, idx, if_persona, interaction_mode="single_turn"):
        self.if_persona = if_persona
        self.interaction_mode = interaction_mode

    def structured_observation(self):
        return {
            "observation_version": "shopping-observation-v7",
            "page_type": "search_home",
            "search_available": True,
            "actions": [],
        }


class ShopAgentParsingTest(unittest.TestCase):
    def test_accepts_direct_action_label(self):
        self.assertEqual(
            _extract_action_from_response("Action: search[乳胶枕]"),
            "search[乳胶枕]",
        )

    def test_accepts_thought_action_format_and_whitespace(self):
        response = "Thought: 先搜索。\n  Action : click[buy now]  "
        self.assertEqual(
            _extract_action_from_response(response), "click[buy now]"
        )

    def test_multiple_action_lines_are_preserved_and_rejected(self):
        response = "Action: search[鞋]\nAction: click[buy now]"
        extracted = _extract_action_from_response(response)
        self.assertEqual(extracted, response)
        with self.assertRaises(ActionParseError) as raised:
            parse_action(extracted)
        self.assertEqual(raised.exception.code, "multiple_actions_not_allowed")

    def test_action_parser_rejects_chained_or_trailing_actions(self):
        with self.assertRaises(ActionParseError) as raised:
            parse_action("click[黑色], click[大号]")
        self.assertEqual(raised.exception.code, "multiple_actions_not_allowed")
        with self.assertRaises(ActionParseError):
            parse_action("click[黑色] trailing")

    def test_reset_hides_private_goal_by_default(self):
        result = _handle_reset_action(_DummyEnv(), 0, 0, if_persona=True)
        self.assertNotIn("private_task_instruction", result)
        self.assertNotIn("goal_options", result)
        self.assertNotIn("reason_key", result)
        self.assertNotIn("__reasoning__", result["user_persona"])
        self.assertEqual(result["task_instruction"], "简化目标")
        self.assertTrue(result["observation"].startswith("页面：搜索首页"))
        self.assertNotIn("[SHOPPING_OBSERVATION", result["observation"])
        self.assertNotIn("search[", result["observation"])
        self.assertNotIn("instruction", result)
        self.assertNotIn("instruction_simple", result)

    def test_private_goal_is_opt_in_for_shopper_simulator(self):
        result = _handle_reset_action(
            _DummyEnv(),
            0,
            0,
            if_persona=True,
            include_private_goal=True,
        )
        self.assertEqual(
            result["private_task_instruction"], "完整隐藏目标"
        )
        self.assertEqual(result["goal_options"], ["黑色"])


if __name__ == "__main__":
    unittest.main()
