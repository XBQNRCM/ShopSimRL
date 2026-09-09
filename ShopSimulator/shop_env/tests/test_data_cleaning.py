import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from data_pipeline.dataset_builder import (
    DatasetBuildError,
    build_clean_corpus,
    full_instruction_task,
    persona_annotation_tasks,
    resolve_persona_annotations,
    source_tasks,
    write_clean_artifacts,
)
from data_pipeline.price_annotation import (
    ANNOTATION_VERSION,
    AnnotationError,
    APPROXIMATE_UPPER_MULTIPLIER,
    request_payload,
    validate_annotation,
)
from scripts.build_clean_dataset import (
    AdaptiveRateLimiter,
    NonRetryableAPIError,
    RateLimitExhaustedError,
    annotate_phase,
    call_task,
)


def product(asin, instruction, *, options=None, simple=None):
    result = {
        "asin": asin,
        "title": f"product {asin}",
        "category": "test",
        "pricing": [10.0],
        "customization_options": options or {},
        "instructions": [
            {
                "instruction": instruction,
                "instruction_simple": simple,
                "attributes": ["test"],
                "instruction_options": [],
            }
        ],
    }
    if simple:
        result["user_persona"] = {"preference": "test"}
    return result


def annotation(kind="no_budget", amount=None, evidence="", source="instruction_simple"):
    price_upper = amount
    if kind == "approximate_upper":
        price_upper = round(amount * APPROXIMATE_UPPER_MULTIPLIER, 8)
    return {
        "kind": kind,
        "amount": amount,
        "evidence": evidence,
        "source_field": source,
        "price_upper": price_upper,
        "annotation_version": ANNOTATION_VERSION,
        "approximate_upper_multiplier": (
            APPROXIMATE_UPPER_MULTIPLIER
            if kind == "approximate_upper"
            else None
        ),
    }


class DataCleaningTest(unittest.TestCase):
    @patch("scripts.build_clean_dataset.requests.post")
    def test_authentication_failure_is_not_retried(self, post):
        response = Mock(status_code=401, headers={})
        post.return_value = response
        tasks, _ = persona_annotation_tasks(
            [product("a", "完整预算100元", simple="预算100元")]
        )

        with self.assertRaisesRegex(DatasetBuildError, "HTTP 401"):
            call_task(
                tasks[0],
                api_key="sk-test-key-long-enough",
                api_url="https://example.invalid/v1/chat/completions",
                model="test-model",
                timeout=1,
                retries=5,
            )

        self.assertEqual(post.call_count, 1)

    def test_request_contains_one_instruction_and_expects_one_object(self):
        task = persona_annotation_tasks(
            [product("a", "完整预算100元", simple="预算100元")]
        )[0][0]
        payload = request_payload(task, "test-model")
        user_input = json.loads(payload["messages"][1]["content"])
        schema = payload["response_format"]["json_schema"]["schema"]

        self.assertIsInstance(user_input, dict)
        self.assertEqual(user_input, {"task_id": 0, "text": "预算100元"})
        self.assertIn("task_id", schema["properties"])
        self.assertNotIn("results", schema["properties"])

    @patch("scripts.build_clean_dataset.requests.post")
    def test_tpm_limit_uses_shared_cooldown_then_retries(self, post):
        task = persona_annotation_tasks(
            [product("a", "完整预算100元", simple="预算100元")]
        )[0][0]
        limited = Mock(
            status_code=429,
            headers={},
            text='{"code":50602,"message":"TPM limit reached"}',
        )
        success = Mock(status_code=200, headers={})
        success.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "task_id": 0,
                                "kind": "hard_upper",
                                "amount": 100,
                                "evidence": "预算100元",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        post.side_effect = [limited, success]
        limiter = Mock()
        limiter.on_rate_limit.return_value = {
            "new_wave": False,
            "cooldown_seconds": 65.0,
            "requests_per_minute": 32.0,
        }

        parsed, _ = call_task(
            task,
            api_key="sk-test-key-long-enough",
            api_url="https://example.invalid/v1/chat/completions",
            model="test-model",
            timeout=1,
            retries=0,
            rate_limit_retries=1,
            rate_limiter=limiter,
        )

        self.assertEqual(parsed["price_upper"], 100.0)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(limiter.wait_for_slot.call_count, 2)
        limiter.on_rate_limit.assert_called_once_with(limited)

    def test_rate_limit_waves_share_cooldown_and_reduce_pacing(self):
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        limiter = AdaptiveRateLimiter(
            requests_per_minute=60,
            cooldown_seconds=65,
            clock=lambda: now[0],
            sleeper=sleep,
        )
        response = Mock(headers={}, text="TPM limit reached")

        first = limiter.on_rate_limit(response)
        same_wave = limiter.on_rate_limit(response)
        now[0] = 66.0
        second_wave = limiter.on_rate_limit(response)

        self.assertTrue(first["new_wave"])
        self.assertEqual(first["requests_per_minute"], 32.0)
        self.assertFalse(same_wave["new_wave"])
        self.assertEqual(same_wave["requests_per_minute"], 32.0)
        self.assertTrue(second_wave["new_wave"])
        self.assertEqual(second_wave["requests_per_minute"], 16.0)

    @patch("scripts.build_clean_dataset.requests.post")
    def test_daily_token_limit_stops_without_retrying(self, post):
        task = persona_annotation_tasks(
            [product("a", "完整预算100元", simple="预算100元")]
        )[0][0]
        post.return_value = Mock(
            status_code=429,
            headers={},
            text='{"message":"TPD limit reached"}',
        )

        with self.assertRaisesRegex(NonRetryableAPIError, "TPD"):
            call_task(
                task,
                api_key="sk-test-key-long-enough",
                api_url="https://example.invalid/v1/chat/completions",
                model="test-model",
                timeout=1,
                retries=5,
            )

        self.assertEqual(post.call_count, 1)

    @patch("scripts.build_clean_dataset.requests.post")
    def test_exhausted_minute_limit_is_a_phase_stopping_error(self, post):
        task = persona_annotation_tasks(
            [product("a", "完整预算100元", simple="预算100元")]
        )[0][0]
        post.return_value = Mock(
            status_code=429,
            headers={},
            text='{"message":"TPM limit reached"}',
        )

        with self.assertRaises(RateLimitExhaustedError):
            call_task(
                task,
                api_key="sk-test-key-long-enough",
                api_url="https://example.invalid/v1/chat/completions",
                model="test-model",
                timeout=1,
                retries=0,
                rate_limit_retries=0,
            )

        self.assertEqual(post.call_count, 1)

    def test_concurrent_results_are_appended_one_line_at_a_time(self):
        products = [
            product(str(index), f"完整需求{index}", simple=f"无预算需求{index}")
            for index in range(3)
        ]
        tasks, _ = persona_annotation_tasks(products)
        release_slow_task = threading.Event()

        def fake_call(task, **_kwargs):
            if task["task_id"] == 1:
                release_slow_task.wait(timeout=2)
                time.sleep(0.05)
            elif task["task_id"] == 2:
                release_slow_task.set()
            return annotation(source=task["source_field"]), {}

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "annotations.jsonl"
            cache = {}
            with patch(
                "scripts.build_clean_dataset.call_task", side_effect=fake_call
            ):
                requested = annotate_phase(
                    tasks,
                    cache_path=cache_path,
                    cache=cache,
                    api_key="sk-test-key-long-enough",
                    api_url="https://example.invalid/v1/chat/completions",
                    model="test-model",
                    workers=2,
                    timeout=1,
                    retries=0,
                    rate_limit_retries=0,
                    rate_limiter=AdaptiveRateLimiter(
                        requests_per_minute=1_000_000,
                        cooldown_seconds=1,
                    ),
                    max_requests=None,
                    source_sha256="source-sha",
                )

            records = [
                json.loads(line)
                for line in cache_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(requested, 3)
        self.assertEqual([record["task_id"] for record in records], [0, 2, 1])
        self.assertEqual(len(cache), 3)

    def test_approximate_budget_is_derived_by_code(self):
        task = persona_annotation_tasks(
            [product("a", "完整需求", simple="预算100元左右")]
        )[0][0]
        parsed = validate_annotation(
            {
                "task_id": 0,
                "kind": "approximate_upper",
                "amount": 100,
                "evidence": "预算100元左右",
            },
            task,
        )
        self.assertEqual(parsed["amount"], 100.0)
        self.assertEqual(parsed["price_upper"], 110.0)

    def test_simple_first_and_full_fallback_are_resolved_by_code(self):
        tasks, _ = persona_annotation_tasks(
            [product("a", "完整指令预算200元以内", simple="想买测试商品")]
        )
        complete = full_instruction_task(tasks[0])
        cache = {
            (0, "instruction_simple"): annotation(),
            (0, "instruction"): annotation(
                "hard_upper", 200.0, "预算200元以内", "instruction"
            ),
        }
        resolved = resolve_persona_annotations(tasks, cache)
        self.assertEqual(resolved[0]["price_upper"], 200.0)
        self.assertEqual(complete["source_field"], "instruction")

    def test_explicit_missing_annotation_uses_auditable_runtime_fallback(self):
        tasks, _ = persona_annotation_tasks(
            [product("a", "完整指令价格150元左右", simple="价格150元左右")]
        )

        with self.assertRaisesRegex(DatasetBuildError, "0:instruction_simple"):
            resolve_persona_annotations(tasks, {})

        resolved = resolve_persona_annotations(
            tasks,
            {},
            runtime_fallback_task_ids={0},
        )
        cleaned, _, stats, audit = build_clean_corpus(
            [product("a", "完整指令价格150元左右", simple="价格150元左右")],
            resolved,
        )

        self.assertIsNone(cleaned[0]["instructions"][0]["price_upper"])
        self.assertEqual(audit[0]["kind"], "runtime_regex_fallback")
        self.assertEqual(audit[0]["missing_source_field"], "instruction_simple")
        self.assertEqual(stats["persona_llm_task_count"], 0)
        self.assertEqual(stats["persona_runtime_regex_task_count"], 1)
        self.assertEqual(stats["persona_runtime_regex_task_ids"], [0])

    def test_filter_precedes_persona_annotation_and_ids_keep_gaps(self):
        products = [
            product("a", "标准任务"),
            product(
                "b",
                "完整persona任务",
                simple="预算100元左右",
                options={
                    "color": [
                        {"value": "red", "price": 10},
                        {"value": "blue", "price": 20},
                    ],
                    "size": [
                        {"value": "s", "price": 10},
                        {"value": "m", "price": 30},
                    ],
                },
            ),
            product("c", "完整persona任务", simple="预算100元左右"),
        ]
        persona_tasks, exclusions = persona_annotation_tasks(products)
        self.assertEqual([task["task_id"] for task in persona_tasks], [2])
        self.assertEqual(exclusions[0]["task_ids"], [1])

        cleaned, _, stats, audit = build_clean_corpus(
            products,
            {2: annotation("approximate_upper", 100.0, "预算100元左右")},
        )
        self.assertEqual(
            [row["instructions"][0]["task_id"] for row in cleaned], [0, 2]
        )
        self.assertIsNone(cleaned[0]["instructions"][0]["price_upper"])
        self.assertEqual(cleaned[1]["instructions"][0]["price_upper"], 110.0)
        self.assertEqual(stats["standard_regex_task_count"], 1)
        self.assertEqual(stats["persona_llm_task_count"], 1)
        self.assertEqual(audit[0]["task_id"], 2)
        self.assertEqual(source_tasks(products)[2]["source_product_index"], 2)

    def test_validation_requires_verbatim_evidence(self):
        task = persona_annotation_tasks(
            [product("a", "完整需求", simple="预算100元以内")]
        )[0][0]
        with self.assertRaises(AnnotationError):
            validate_annotation(
                {
                    "task_id": 0,
                    "kind": "hard_upper",
                    "amount": 100,
                    "evidence": "预算200元以内",
                },
                task,
            )

    def test_writes_new_archive_without_touching_source(self):
        products = [product("a", "完整需求", simple="预算100元以内")]
        annotations = {0: annotation("hard_upper", 100.0, "预算100元以内")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.json.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                json.dump(products, handle, ensure_ascii=False)
            original = source.read_bytes()

            manifest = write_clean_artifacts(
                source=source,
                output=root / "cleaned.json.gz",
                manifest_path=root / "manifest.json",
                exclusions_path=root / "exclusions.json",
                splits_path=root / "splits.json",
                audit_path=root / "audit.json",
                products=products,
                persona_annotations=annotations,
                model="test-model",
                api_url="https://example.invalid/v1/chat/completions",
            )

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(manifest["persona_llm_task_count"], 1)
            with gzip.open(root / "cleaned.json.gz", "rb") as handle:
                output_data_sha = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(manifest["output_data_sha256"], output_data_sha)
            with gzip.open(root / "cleaned.json.gz", "rt", encoding="utf-8") as handle:
                cleaned = json.load(handle)
            self.assertEqual(cleaned[0]["instructions"][0]["task_id"], 0)
            self.assertNotIn("price_constraint", cleaned[0]["instructions"][0])


if __name__ == "__main__":
    unittest.main()
