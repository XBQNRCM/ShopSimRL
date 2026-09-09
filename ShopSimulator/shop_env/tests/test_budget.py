import unittest

from web_agent_site.engine.budget import explicit_budget_from_instruction
from web_agent_site.engine.goal import get_goals


class InstructionBudgetTest(unittest.TestCase):
    def test_explicit_ceiling_range_and_approximate_budget(self):
        self.assertEqual(explicit_budget_from_instruction("预算在1000元以下。"), 1000.0)
        self.assertEqual(explicit_budget_from_instruction("价格不超过 2199 元"), 2199.0)
        self.assertEqual(explicit_budget_from_instruction("价格在70元左右"), 77.0)
        self.assertEqual(explicit_budget_from_instruction("价格在130-140元之间"), 140.0)
        self.assertEqual(explicit_budget_from_instruction("价格30元到40元之间"), 40.0)
        self.assertEqual(explicit_budget_from_instruction("价格在10-20之间"), 20.0)
        self.assertIsNone(explicit_budget_from_instruction("预算4k+"))
        self.assertEqual(explicit_budget_from_instruction("预算在1万元左右"), 11000.0)
        self.assertEqual(explicit_budget_from_instruction("预算1万2以内"), 12000.0)
        self.assertEqual(explicit_budget_from_instruction("差不多30元左右"), 33.0)
        self.assertEqual(explicit_budget_from_instruction("预算七十元上下"), 77.0)
        self.assertEqual(explicit_budget_from_instruction("价格别超过160元"), 160.0)
        self.assertEqual(
            explicit_budget_from_instruction("预期价格在350元和400元之间"),
            400.0,
        )
        self.assertEqual(explicit_budget_from_instruction("预算是17000元"), 17000.0)
        self.assertEqual(explicit_budget_from_instruction("预算不超过一千八"), 1800.0)
        self.assertEqual(explicit_budget_from_instruction("预算就二十来块钱"), 22.0)
        self.assertEqual(explicit_budget_from_instruction("五元以内"), 5.0)
        self.assertIsNone(explicit_budget_from_instruction("价格要超过20元"))

    def test_missing_budget_does_not_use_gold_product_price(self):
        product = {
            "asin": "1",
            "category": "家居›枕头",
            "title": "测试枕头",
            "pricing": [9999],
            "instructions": [
                {
                    "instruction": "买一个柔软的枕头",
                    "attributes": ["柔软"],
                    "instruction_options": [],
                }
            ],
        }
        goal = get_goals([product], {"1": 9999.0})[0]
        self.assertIsNone(goal["price_upper"])

    def test_precomputed_budget_takes_priority_over_runtime_parser(self):
        product = {
            "asin": "1",
            "category": "家居›枕头",
            "title": "测试枕头",
            "pricing": [99],
            "instructions": [
                {
                    "task_id": 1479,
                    "task_uid": "shopsim-test",
                    "instruction": "预算不超过900元",
                    "instruction_simple": "预算不超过900元",
                    "attributes": ["柔软"],
                    "instruction_options": [],
                    "price_upper": 100.0,
                    "price_constraint": {"status": "valid"},
                }
            ],
        }

        goal = get_goals([product])[0]

        self.assertEqual(goal["task_id"], 1479)
        self.assertEqual(goal["task_uid"], "shopsim-test")
        self.assertEqual(goal["price_upper"], 100.0)

    def test_empty_precomputed_budget_falls_back_to_runtime_parser(self):
        product = {
            "asin": "1",
            "category": "家居›枕头",
            "title": "测试枕头",
            "pricing": [99],
            "instructions": [
                {
                    "instruction": "预算不超过300元，买一个柔软的枕头",
                    "instruction_simple": None,
                    "attributes": ["柔软"],
                    "instruction_options": [],
                    "price_upper": None,
                }
            ],
        }

        goal = get_goals([product])[0]

        self.assertEqual(goal["price_upper"], 300.0)


if __name__ == "__main__":
    unittest.main()
