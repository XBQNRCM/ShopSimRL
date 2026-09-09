import unittest

from shop_env.pack_api import app


class DebugUiTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_agent_view_and_versioned_assets_are_served(self):
        page = self.client.get("/debug-ui")
        try:
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.content_type)
            self.assertIn("ShopSimulator", page.get_data(as_text=True))
            self.assertIn("Agent View", page.get_data(as_text=True))
        finally:
            page.close()

        script = self.client.get("/debug-ui/app.js")
        try:
            self.assertEqual(script.status_code, 200)
            self.assertIn("javascript", script.content_type)
            self.assertIn("/api/shop_agent", script.get_data(as_text=True))
            self.assertIn("action_feedback", script.get_data(as_text=True))
            self.assertIn("result.observation", script.get_data(as_text=True))
            self.assertIn("task_instruction", script.get_data(as_text=True))
            self.assertNotIn("result.instruction", script.get_data(as_text=True))
        finally:
            script.close()

        stylesheet = self.client.get("/debug-ui/styles.css")
        try:
            self.assertEqual(stylesheet.status_code, 200)
            self.assertIn("text/css", stylesheet.content_type)
        finally:
            stylesheet.close()

    def test_replay_ui_and_assets_are_served(self):
        page = self.client.get("/replay-ui")
        try:
            self.assertEqual(page.status_code, 200)
            text = page.get_data(as_text=True)
            self.assertIn("Experiment Replay", text)
            self.assertIn("任务轨迹", text)
            self.assertIn("outcomeComparison", text)
            self.assertIn("/debug-ui/replay.js", text)
        finally:
            page.close()

        script = self.client.get("/debug-ui/replay.js")
        try:
            self.assertEqual(script.status_code, 200)
            text = script.get_data(as_text=True)
            self.assertIn("/api/replay/run", text)
            self.assertIn("/api/replay/episode", text)
            self.assertIn("目标商品 vs 实际购买", text)
            self.assertIn("Gold truth option", text)
        finally:
            script.close()

        stylesheet = self.client.get("/debug-ui/replay.css")
        try:
            self.assertEqual(stylesheet.status_code, 200)
            self.assertIn("text/css", stylesheet.content_type)
        finally:
            stylesheet.close()

    def test_health_check_advertises_debug_contract(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        health = response.get_json()
        self.assertEqual(health["debug_ui"], "/debug-ui")
        self.assertEqual(health["replay_ui"], "/replay-ui")
        self.assertEqual(health["primary_reward"], "r_strict")
        self.assertEqual(
            health["observation_version"], "shopping-observation-v7"
        )
        self.assertEqual(
            health["environment_version"], "shopsimulator-paper-aligned-v8"
        )


if __name__ == "__main__":
    unittest.main()
