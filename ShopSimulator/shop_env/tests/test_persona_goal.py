import unittest

from web_agent_site.engine.goal import get_goals


class PersonaGoalContractTest(unittest.TestCase):
    def test_persona_uses_simple_instruction_without_instruction_sample(self):
        products = [
            {
                "asin": "000000000001",
                "category": "电子产品 › 耳机",
                "title": "测试耳机",
                "user_persona": {"预算偏好": "节省"},
                "instructions": [
                    {
                        "instruction": "买黑色降噪耳机，预算不超过500元",
                        "instruction_simple": "买一副耳机",
                        "attributes": ["黑色", "降噪"],
                        "instruction_options": ["黑色"],
                    }
                ],
            }
        ]
        goal = get_goals(
            products, {"000000000001": 399.0}, if_persona=True
        )[0]
        self.assertEqual(
            goal["instruction_text"],
            products[0]["instructions"][0]["instruction"],
        )
        self.assertEqual(goal["instruction_simple"], "买一副耳机")
        self.assertEqual(goal["user_persona"], {"预算偏好": "节省"})
        self.assertEqual(goal["price_upper"], 500.0)
        self.assertNotIn("query", goal)

    def test_missing_simple_instruction_falls_back_to_full_instruction(self):
        products = [
            {
                "asin": "000000000002",
                "category": "家居 › 灯",
                "title": "测试台灯",
                "instructions": [
                    {
                        "instruction": "买一盏台灯",
                        "attributes": ["台灯"],
                        "instruction_options": [],
                    }
                ],
            }
        ]
        goal = get_goals(products, {"000000000002": 99.0})[0]
        self.assertEqual(goal["instruction_simple"], "买一盏台灯")
        self.assertEqual(goal["user_persona"], {})
        self.assertIsNone(goal["reason_key"])

    def test_simple_instruction_budget_takes_priority_over_full_instruction(self):
        product = {
            "asin": "000000000003",
            "category": "家居 › 灯",
            "title": "预算测试台灯",
            "instructions": [
                {
                    "instruction": "买一盏预算不超过500元的台灯",
                    "instruction_simple": "找一盏价格在200元以内的台灯",
                    "attributes": ["台灯"],
                    "instruction_options": [],
                }
            ],
        }
        goal = get_goals([product])[0]
        self.assertEqual(goal["price_upper"], 200.0)


if __name__ == "__main__":
    unittest.main()
