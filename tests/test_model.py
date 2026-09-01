from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import requests

from shopsimrl.model import (
    ModelRequestError,
    OpenAICompatibleChatModel,
    OpenAICompatibleConfig,
)


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text
        self.reason = text
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._body


class FakeStreamResponse(FakeResponse):
    def __init__(self, events=None, *, error=None):
        super().__init__(200)
        self.events = events or []
        self.error = error
        self.closed = False

    def iter_lines(self, decode_unicode=False):
        for event in self.events:
            yield f"data: {json.dumps(event)}"
        if self.error is not None:
            raise self.error
        yield "data: [DONE]"

    def close(self):
        self.closed = True


class ModelTest(unittest.TestCase):
    @staticmethod
    def _tools():
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object"},
                },
            }
        ]

    def test_retries_throttling_and_preserves_sampling_payload(self):
        success = FakeResponse(
            200,
            {
                "id": "response-1",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "brief reason",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query":"x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"total_tokens": 4},
            },
        )
        config = OpenAICompatibleConfig(
            model="model",
            base_url="http://model/v1",
            api_key_env=None,
            max_retries=2,
            retry_backoff_seconds=0,
            extra_body={"enable_thinking": True},
        )
        model = OpenAICompatibleChatModel(config)
        with patch.object(
            model.session,
            "post",
            side_effect=[FakeResponse(429, text="busy"), success],
        ) as post, patch("shopsimrl.model.time.sleep"):
            output = model.generate(
                [{"role": "user", "content": "go"}],
                seed=9,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        self.assertIsNone(output.content)
        self.assertEqual(output.reasoning, "brief reason")
        self.assertEqual(output.tool_calls[0].name, "search")
        self.assertEqual(output.tool_calls[0].arguments, {"query": "x"})
        self.assertEqual(post.call_count, 2)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["seed"], 9)
        self.assertTrue(payload["enable_thinking"])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["tools"][0]["function"]["name"], "search")

    def test_streaming_assembles_reasoning_and_tool_call_fragments(self):
        response = FakeStreamResponse(
            [
                {
                    "id": "stream-1",
                    "choices": [
                        {
                            "delta": {"reasoning_content": "brief "},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "stream-1",
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "reason",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query"',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "stream-1",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": ':"x"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"total_tokens": 4},
                },
            ]
        )
        config = OpenAICompatibleConfig(
            model="model",
            base_url="http://model/v1",
            api_key_env=None,
            stream=True,
        )
        model = OpenAICompatibleChatModel(config)
        with patch.object(model.session, "post", return_value=response) as post:
            output = model.generate(
                [{"role": "user", "content": "go"}], tools=self._tools()
            )

        self.assertEqual(output.reasoning, "brief reason")
        self.assertEqual(output.tool_calls[0].name, "search")
        self.assertEqual(output.tool_calls[0].arguments, {"query": "x"})
        self.assertEqual(output.finish_reason, "tool_calls")
        self.assertEqual(output.usage, {"total_tokens": 4})
        self.assertTrue(response.closed)
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])

    def test_streaming_retries_an_interrupted_response_from_scratch(self):
        interrupted = FakeStreamResponse(
            [
                {
                    "id": "partial",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "partial-call",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query"',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ],
            error=requests.exceptions.SSLError("unexpected eof"),
        )
        complete = FakeStreamResponse(
            [
                {
                    "id": "complete",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-2",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query":"recovered"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            ]
        )
        config = OpenAICompatibleConfig(
            model="model",
            base_url="http://model/v1",
            api_key_env=None,
            stream=True,
            max_retries=2,
            retry_backoff_seconds=0,
        )
        model = OpenAICompatibleChatModel(config)
        with patch.object(
            model.session, "post", side_effect=[interrupted, complete]
        ) as post, patch("shopsimrl.model.time.sleep"):
            output = model.generate(
                [{"role": "user", "content": "go"}], tools=self._tools()
            )

        self.assertEqual(post.call_count, 2)
        self.assertTrue(interrupted.closed)
        self.assertTrue(complete.closed)
        self.assertEqual(output.response_id, "complete")
        self.assertEqual(output.tool_calls[0].arguments, {"query": "recovered"})

    def test_streaming_synthesizes_a_missing_tool_call_id(self):
        response = FakeStreamResponse(
            [
                {
                    "id": "stream-without-call-id",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query":"x"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            ]
        )
        config = OpenAICompatibleConfig(
            model="model",
            base_url="http://model/v1",
            api_key_env=None,
            stream=True,
        )
        model = OpenAICompatibleChatModel(config)
        with patch.object(model.session, "post", return_value=response):
            output = model.generate(
                [{"role": "user", "content": "go"}], tools=self._tools()
            )

        self.assertTrue(output.tool_calls[0].call_id.startswith("call-stream-"))
        self.assertEqual(output.tool_calls[0].name, "search")
        self.assertEqual(output.tool_calls[0].arguments, {"query": "x"})

    def test_non_retryable_http_error_fails_immediately(self):
        config = OpenAICompatibleConfig(
            model="model",
            base_url="http://model/v1",
            api_key_env=None,
            max_retries=5,
        )
        model = OpenAICompatibleChatModel(config)
        with patch.object(
            model.session,
            "post",
            return_value=FakeResponse(400, text="bad request"),
        ) as post:
            with self.assertRaises(ModelRequestError):
                model.generate([])
        self.assertEqual(post.call_count, 1)

    def test_protocol_error_is_returned_without_resampling(self):
        config = OpenAICompatibleConfig(
            model="model",
            base_url="http://model/v1",
            api_key_env=None,
            max_retries=5,
        )
        model = OpenAICompatibleChatModel(config)
        response = FakeResponse(
            200,
            {
                "id": "bad-response",
                "choices": [
                    {
                        "message": {
                            "content": "Action: click[buy now]",
                            "reasoning_content": "done",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
        with patch.object(model.session, "post", return_value=response) as post:
            output = model.generate(
                [{"role": "user", "content": "go"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "click",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        self.assertEqual(post.call_count, 1)
        self.assertEqual(output.content, "Action: click[buy now]")
        self.assertEqual(output.protocol_error["code"], "tool_call_count")
        self.assertEqual(
            output.raw_response["choices"][0]["message"]["content"],
            "Action: click[buy now]",
        )

    def test_text_fallback_is_a_protocol_error_when_tool_call_is_required(self):
        output = OpenAICompatibleChatModel._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Action: click[buy now]",
                            "reasoning_content": "done",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            require_tool_call=True,
            allowed_tools={"click"},
        )
        self.assertEqual(output["protocol_error"]["code"], "tool_call_count")
        self.assertEqual(output["content"], "Action: click[buy now]")

    def test_tool_call_with_visible_content_is_a_protocol_error(self):
        output = OpenAICompatibleChatModel._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": "I will search now.",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"query":"x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            require_tool_call=True,
            allowed_tools={"search"},
        )
        self.assertEqual(output["protocol_error"]["code"], "mixed_content")
        self.assertEqual(output["tool_calls"][0].name, "search")

    def test_length_before_tool_call_is_a_policy_failure(self):
        output = OpenAICompatibleChatModel._parse_response(
            {
                "id": "truncated-response",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "unfinished reasoning",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 2048},
            },
            require_tool_call=True,
            allowed_tools={"search"},
        )

        self.assertIsNone(output["protocol_error"])
        self.assertEqual(output["policy_failure"]["code"], "generation_length")
        self.assertEqual(output["reasoning"], "unfinished reasoning")
        self.assertEqual(output["raw_response"]["id"], "truncated-response")


if __name__ == "__main__":
    unittest.main()
