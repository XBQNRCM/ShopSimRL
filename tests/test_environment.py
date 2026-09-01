import unittest

from shopsimrl.environment import ShopSimulatorConfig, ShopSimulatorHTTPEnvironment


class _Response:
    status_code = 200
    ok = True
    reason = "OK"

    def __init__(self, result):
        self._result = result

    def json(self):
        return {"result": self._result}


class _Session:
    def __init__(self):
        self.calls = []
        self.trust_env = False

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "payload": json, "timeout": timeout})
        result = {
            "observation": "page",
            "observation_state": {"observation_version": "test-v1"},
        }
        if json["action"] == "reset":
            result.update(
                {
                    "task_instruction": "task",
                    "env_idx": 0,
                    "lease_id": "lease",
                }
            )
        elif json["action"] == "terminate":
            result.update(
                {
                    "done": True,
                    "reward": 0.0,
                    "termination_reason": json["termination_reason"],
                }
            )
        return _Response(result)

    def close(self):
        pass


class EnvironmentClientTest(unittest.TestCase):
    def test_reset_waits_for_a_slot_without_disabling_other_request_timeouts(self):
        environment = ShopSimulatorHTTPEnvironment(
            ShopSimulatorConfig(timeout=12.0)
        )
        session = _Session()
        environment.session = session

        environment.reset(7)
        environment.step("search[query]")
        environment.terminate("generation_length")

        self.assertEqual(session.calls[0]["timeout"], (12.0, None))
        self.assertEqual(session.calls[1]["timeout"], 12.0)
        self.assertEqual(session.calls[2]["timeout"], 12.0)
        self.assertEqual(
            session.calls[2]["payload"]["termination_reason"],
            "generation_length",
        )


if __name__ == "__main__":
    unittest.main()
