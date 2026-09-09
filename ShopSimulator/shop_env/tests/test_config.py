import json
from pathlib import Path
import unittest

from web_agent_site.engine.config import load_config, validate_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "environment.json"


class PaperAlignedConfigTest(unittest.TestCase):
    def test_repository_config_matches_paper_contract(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["environment_version"], "shopsimulator-paper-aligned-v8"
        )
        self.assertEqual(
            config["search"]["version"], "shopsimulator-multifield-bm25-v3"
        )
        self.assertEqual(
            config["search"]["option_document_version"], "public-axis-value-v1"
        )
        self.assertEqual(config["reward"]["version"], "shopsimulator-paper-reward-v2")
        self.assertEqual(config["reward"]["primary"], "r_strict")
        self.assertEqual(config["episode_limits"], {"single_turn": 30, "multi_turn": 40})
        self.assertEqual(config["catalog_filter"]["max_price_affecting_axes"], 1)
        self.assertEqual(config["variant_price_version"], "variant-price-paper-v2")
        self.assertEqual(config["observation_version"], "shopping-observation-v7")
        self.assertEqual(config["tool_version"], "shopping-tools-paper-v4")
        self.assertNotIn("termination", config)
        self.assertNotIn("reward_feature_version", config)

    def test_reward_and_limit_drift_are_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["reward"]["primary"] = "r_loose"
        with self.assertRaisesRegex(ValueError, "primary reward"):
            validate_config(config)

        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["episode_limits"]["single_turn"] = 35
        with self.assertRaisesRegex(ValueError, "30"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
