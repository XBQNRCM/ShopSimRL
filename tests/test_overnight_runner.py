from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_trace2skill_overnight as overnight


def _write_gate_a_manifest(
    path: Path,
    *,
    status: str,
    requested: int = 2,
    completed: int = 2,
    failed: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "summary": {
                    "counts": {
                        "requested": requested,
                        "completed": completed,
                        "failed": failed,
                        "coverage": completed / requested,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


class OvernightRunnerTest(unittest.TestCase):
    def test_gate_a_retries_incomplete_run_and_resumes_until_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate-a"
            manifest_path = output_dir / "gate_a_manifest.json"
            calls = 0

            def fake_run_cli(arguments):
                nonlocal calls
                self.assertEqual(arguments[:2], ["trace2skill", "gate-a"])
                calls += 1
                if calls == 1:
                    _write_gate_a_manifest(
                        manifest_path,
                        status="incomplete",
                        completed=1,
                        failed=1,
                    )
                    return 1
                _write_gate_a_manifest(manifest_path, status="complete")
                (output_dir / "selected_skillbank.json").write_text(
                    "{}", encoding="utf-8"
                )
                return 0

            spec = SimpleNamespace(output_dir=output_dir)
            with (
                patch.object(
                    overnight,
                    "load_trace2skill_evaluation_config",
                    return_value=spec,
                ),
                patch.object(overnight, "_run_cli", side_effect=fake_run_cli),
                patch.object(overnight.time, "sleep") as sleep,
            ):
                selected = overnight.run_gate_a_until_complete(
                    Path("gate-a.yaml"),
                    max_retries=5,
                    retry_delay_seconds=3,
                )

            self.assertEqual(selected, output_dir / "selected_skillbank.json")
            self.assertEqual(calls, 2)
            sleep.assert_called_once_with(3)

    def test_gate_a_stops_after_initial_run_plus_max_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "gate-a"
            manifest_path = output_dir / "gate_a_manifest.json"

            def fake_run_cli(_arguments):
                _write_gate_a_manifest(
                    manifest_path,
                    status="incomplete",
                    completed=1,
                    failed=1,
                )
                return 1

            spec = SimpleNamespace(output_dir=output_dir)
            with (
                patch.object(
                    overnight,
                    "load_trace2skill_evaluation_config",
                    return_value=spec,
                ),
                patch.object(
                    overnight, "_run_cli", side_effect=fake_run_cli
                ) as run_cli,
            ):
                with self.assertRaisesRegex(
                    overnight.OvernightRunError, "after 3 attempts"
                ):
                    overnight.run_gate_a_until_complete(
                        Path("gate-a.yaml"),
                        max_retries=2,
                        retry_delay_seconds=0,
                    )

            self.assertEqual(run_cli.call_count, 3)

    def test_main_runs_gate_a_before_equipped_test(self):
        order: list[str] = []

        def fake_gate_a(*_args, **_kwargs):
            order.append("gate-a")
            return Path("selected_skillbank.json")

        def fake_equipped(*_args, **_kwargs):
            order.append("equipped")
            return Path("equipped-run")

        with (
            patch.object(overnight, "run_gate_a_until_complete", fake_gate_a),
            patch.object(overnight, "run_equipped_test", fake_equipped),
        ):
            returncode = overnight.main([])

        self.assertEqual(returncode, 0)
        self.assertEqual(order, ["gate-a", "equipped"])

    def test_equipped_test_uses_generic_plan_and_run_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            run_dir = output_dir / "equipped"
            config_path = Path("equipped.yaml")
            commands: list[list[object]] = []

            def fake_run_cli(arguments):
                commands.append(list(arguments))
                if arguments[0] == "run":
                    run_dir.mkdir(parents=True)
                    (run_dir / "summary.json").write_text(
                        json.dumps(
                            {
                                "counts": {
                                    "requested": 2,
                                    "completed": 2,
                                    "failed": 0,
                                    "coverage": 1.0,
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                return 0

            spec = SimpleNamespace(output_dir=output_dir, name="equipped")
            with (
                patch.object(
                    overnight, "load_experiment_config", return_value=spec
                ),
                patch.object(overnight, "_run_cli", side_effect=fake_run_cli),
            ):
                result = overnight.run_equipped_test(config_path)

            self.assertEqual(result, run_dir)
            self.assertEqual(
                commands,
                [["plan", str(config_path)], ["run", str(config_path)]],
            )


if __name__ == "__main__":
    unittest.main()
