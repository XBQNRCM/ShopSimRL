from __future__ import annotations

import json
import unittest

from shopsimrl.prompts import ShoppingPromptBuilder
from shopsimrl.runtime import AgentRuntime, RuntimeConfig
from shopsimrl.schemas import EpisodeJob, ModelOutput, Skill, ToolCall


class FakeModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.seeds = []
        self.closed = False

    def identity(self):
        return {"model": "fake"}

    def generate(self, messages, *, seed=None, tools=None):
        self.seeds.append(seed)
        self.tools = getattr(self, "tools", []) + [tools]
        item = next(self.actions)
        if isinstance(item, ModelOutput):
            return item
        name, arguments = item
        return ModelOutput(
            reasoning="private reasoning",
            tool_calls=(
                ToolCall(
                    call_id=f"call-{len(self.seeds)}",
                    name=name,
                    arguments=arguments,
                    raw_arguments=json.dumps(arguments),
                ),
            ),
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )

    def close(self):
        self.closed = True


class FakeEnvironment:
    def __init__(self, done_after=2):
        self.done_after = done_after
        self.actions = []
        self.terminated = False
        self.termination_reasons = []
        self.closed = False

    def identity(self):
        return {
            "environment_version": "fake-v1",
            "observation_version": "obs-v1",
        }

    def reset(self, task_id):
        return {
            "task_instruction": f"buy task {task_id}",
            "observation": "home",
            "observation_state": {
                "observation_version": "obs-v1",
                "search_available": True,
                "actions": [],
            },
            "user_persona": {"style": "simple"},
            "task_mode": "persona",
            "env_idx": 7,
            "lease_id": "secret-lease",
        }

    def step(self, action):
        self.actions.append(action)
        done = len(self.actions) >= self.done_after
        return {
            "done": done,
            "observation": f"page-{len(self.actions)}",
            "observation_state": {
                "observation_version": "obs-v1",
                "search_available": not done,
                "actions": [] if done else ["buy now"],
            },
            "reward": 1.0 if done else 0.0,
            "reward_detail": {"r_success": 1, "r_strict": 1.0} if done else {},
            "termination_reason": "purchase" if done else None,
            "action_feedback": {"valid": True},
            "env_idx": 7,
        }

    def terminate(self, reason):
        self.terminated = True
        self.termination_reasons.append(reason)
        return {
            "done": True,
            "observation": "limit",
            "observation_state": {"observation_version": "obs-v1"},
            "reward": 0.0,
            "reward_detail": {"r_success": 0, "r_strict": 0.0},
            "termination_reason": reason,
        }

    def close(self):
        self.closed = True


class OneSkill:
    def identity(self):
        return {"provider": "test"}

    def select(self, context):
        self.context = context
        return [Skill("compare", "Compare requirements before buying.", "2")]


class RuntimeTest(unittest.TestCase):
    def test_completed_trace_is_self_contained_and_hides_lease(self):
        model = FakeModel(
            [("search", {"query": "query"}), ("click", {"value": "buy now"})]
        )
        environment = FakeEnvironment(done_after=2)
        skills = OneSkill()
        runtime = AgentRuntime(
            model=model,
            environment=environment,
            prompt_builder=ShoppingPromptBuilder(system_prompt="shop"),
            skill_provider=skills,
            config=RuntimeConfig(max_steps=3),
        )
        job = EpisodeJob(task_id=12, sample_id=1, seed=100, split="train")

        trace = runtime.run(job)

        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["final"]["reward"], 1.0)
        self.assertEqual(len(trace["steps"]), 2)
        self.assertEqual(model.seeds, [100, 101])
        self.assertEqual(environment.actions, ["search[query]", "click[buy now]"])
        self.assertEqual(trace["steps"][0]["model"]["reasoning"], "private reasoning")
        self.assertEqual(
            [tool["function"]["name"] for tool in model.tools[0]], ["search"]
        )
        self.assertEqual(
            [tool["function"]["name"] for tool in model.tools[1]],
            ["search", "click"],
        )
        self.assertEqual(
            model.tools[1][1]["function"]["parameters"]["properties"]["value"]["enum"],
            ["buy now"],
        )
        self.assertEqual(trace["conversation"][2]["role"], "assistant")
        self.assertEqual(trace["conversation"][3]["role"], "tool")
        self.assertNotIn("lease_id", trace["reset"])
        self.assertNotIn("env_idx", trace["steps"][0]["environment"])
        self.assertIn("<skill", trace["conversation"][0]["content"])
        self.assertEqual(trace["selected_skills"][0]["skill_id"], "compare")
        self.assertTrue(model.closed)
        self.assertTrue(environment.closed)
        self.assertFalse(environment.terminated)

    def test_runtime_uses_environment_terminal_for_action_limit(self):
        model = FakeModel([("search", {"query": "a"}), ("search", {"query": "b"})])
        environment = FakeEnvironment(done_after=99)
        runtime = AgentRuntime(
            model=model,
            environment=environment,
            config=RuntimeConfig(max_steps=2),
        )

        trace = runtime.run(EpisodeJob(1, 0, 5, "eval"))

        self.assertEqual(trace["status"], "completed")
        self.assertTrue(environment.terminated)
        self.assertEqual(environment.termination_reasons, ["action_limit"])
        self.assertEqual(trace["final"]["termination_reason"], "action_limit")

    def test_generation_length_is_a_valid_zero_reward_policy_failure(self):
        truncated = ModelOutput(
            reasoning="long unfinished reasoning",
            finish_reason="length",
            usage={"completion_tokens": 2048},
            policy_failure={
                "code": "generation_length",
                "message": "generation reached max_tokens",
            },
            raw_response={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning_content": "long unfinished reasoning",
                        },
                    }
                ]
            },
        )
        model = FakeModel([truncated])
        environment = FakeEnvironment(done_after=99)
        runtime = AgentRuntime(
            model=model,
            environment=environment,
            config=RuntimeConfig(max_steps=3),
        )

        trace = runtime.run(EpisodeJob(5, 0, 80, "train"))

        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["final"]["reward"], 0.0)
        self.assertEqual(trace["final"]["termination_reason"], "generation_length")
        self.assertEqual(environment.termination_reasons, ["generation_length"])
        self.assertEqual(environment.actions, [])
        self.assertEqual(len(trace["steps"]), 1)
        self.assertEqual(
            trace["steps"][0]["policy_failure"]["code"], "generation_length"
        )
        self.assertEqual(
            trace["steps"][0]["model"]["reasoning"],
            "long unfinished reasoning",
        )
        self.assertIsNone(trace["steps"][0]["action"])
        self.assertIsNone(trace["steps"][0]["environment"])

    def test_protocol_error_is_traced_and_consumes_one_step_before_repair(self):
        bad_output = ModelOutput(
            content="Action: search[query]",
            reasoning="used the legacy format",
            protocol_error={
                "code": "tool_call_count",
                "message": "model must return exactly one tool call, got 0",
            },
            raw_response={
                "choices": [
                    {"message": {"content": "Action: search[query]"}}
                ]
            },
        )
        model = FakeModel(
            [
                bad_output,
                ("search", {"query": "query"}),
                ("click", {"value": "buy now"}),
            ]
        )
        environment = FakeEnvironment(done_after=2)
        runtime = AgentRuntime(
            model=model,
            environment=environment,
            config=RuntimeConfig(max_steps=3),
        )

        trace = runtime.run(EpisodeJob(2, 0, 50, "eval"))

        self.assertEqual(trace["status"], "completed")
        self.assertEqual(model.seeds, [50, 51, 52])
        self.assertEqual(len(trace["steps"]), 3)
        self.assertEqual(environment.actions, ["search[query]", "click[buy now]"])
        first = trace["steps"][0]
        self.assertEqual(first["protocol_error"]["code"], "tool_call_count")
        self.assertEqual(first["model"]["content"], "Action: search[query]")
        self.assertIsNone(first["action"])
        self.assertIsNone(first["environment"])
        self.assertEqual(trace["conversation"][2]["role"], "assistant")
        self.assertEqual(trace["conversation"][3]["role"], "user")
        feedback = json.loads(trace["conversation"][3]["content"])
        self.assertEqual(feedback["type"], "tool_protocol_error")
        self.assertIn("search", feedback["available_tools"])

    def test_invalid_argument_shape_is_repaired_without_environment_action(self):
        model = FakeModel(
            [
                ("search", {"wrong": "query"}),
                ("search", {"query": "query"}),
                ("click", {"value": "buy now"}),
            ]
        )
        environment = FakeEnvironment(done_after=2)
        runtime = AgentRuntime(
            model=model,
            environment=environment,
            config=RuntimeConfig(max_steps=3),
        )

        trace = runtime.run(EpisodeJob(3, 0, 60, "eval"))

        self.assertEqual(trace["status"], "completed")
        self.assertEqual(environment.actions, ["search[query]", "click[buy now]"])
        self.assertEqual(
            trace["steps"][0]["protocol_error"]["code"],
            "invalid_tool_arguments",
        )
        self.assertEqual(trace["conversation"][2]["role"], "assistant")
        self.assertEqual(trace["conversation"][3]["role"], "tool")

    def test_click_enum_violation_is_still_sent_to_environment(self):
        model = FakeModel(
            [("search", {"query": "query"}), ("click", {"value": "92"})]
        )
        environment = FakeEnvironment(done_after=2)
        runtime = AgentRuntime(
            model=model,
            environment=environment,
            config=RuntimeConfig(max_steps=2),
        )

        trace = runtime.run(EpisodeJob(4, 0, 70, "eval"))

        self.assertEqual(trace["status"], "completed")
        self.assertEqual(environment.actions, ["search[query]", "click[92]"])
        self.assertNotIn("protocol_error", trace["steps"][1])


if __name__ == "__main__":
    unittest.main()
