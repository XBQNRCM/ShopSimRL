from types import SimpleNamespace
import unittest

from web_agent_site.engine.goal import empty_paper_reward_detail
from web_agent_site.envs.web_agent_text_env import SimServer


class PaperTerminationTest(unittest.TestCase):
    def test_action_limit_is_a_zero_score_environment_terminal(self):
        server = SimpleNamespace(
            user_sessions={"session": {"goal": {"asin": "gold"}, "done": False}}
        )
        status = SimServer.terminate_action_limit(server, "session")
        self.assertTrue(status["done"])
        self.assertEqual(status["reward"], 0.0)
        self.assertEqual(status["termination_reason"], "action_limit")
        self.assertEqual(status["reward_detail"]["r_loose"], 0.0)
        self.assertEqual(status["reward_detail"]["r_success"], 0)
        self.assertEqual(status["purchase"], {})
        self.assertTrue(server.user_sessions["session"]["done"])

    def test_action_limit_cannot_overwrite_an_existing_purchase_terminal(self):
        server = SimpleNamespace(
            user_sessions={"session": {"goal": {"asin": "gold"}, "done": True}}
        )
        with self.assertRaisesRegex(RuntimeError, "already terminated"):
            SimServer.terminate_action_limit(server, "session")

    def test_generation_length_is_a_zero_score_environment_terminal(self):
        server = SimpleNamespace(
            user_sessions={"session": {"goal": {"asin": "gold"}, "done": False}}
        )
        status = SimServer.terminate_unpurchased(
            server, "session", "generation_length"
        )
        self.assertTrue(status["done"])
        self.assertEqual(status["reward"], 0.0)
        self.assertEqual(status["termination_reason"], "generation_length")
        self.assertEqual(
            status["reward_detail"]["termination_reason"], "generation_length"
        )
        self.assertEqual(status["reward_detail"]["r_success"], 0)

    def test_no_purchase_terminal_has_only_paper_metrics(self):
        detail = empty_paper_reward_detail(termination_reason="action_limit")
        self.assertEqual(detail["termination_reason"], "action_limit")
        self.assertNotIn("reward_type", detail)
        self.assertNotIn("graceful_stop", detail)
        self.assertNotIn("early_abstain", detail)


if __name__ == "__main__":
    unittest.main()
