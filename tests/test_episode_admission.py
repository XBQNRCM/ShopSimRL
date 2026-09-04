import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest

from shopsimrl import slime_runtime
from shopsimrl.curriculum import SkillAssignment


def _episode(args, *, diagnostic=False):
    return slime_runtime._run_episode(
        args, None, {}, skills=(), assignment=None, diagnostic=diagnostic
    )


def test_episode_admission_keeps_thread_pool_available(monkeypatch):
    args = SimpleNamespace(shopsim_episode_concurrency=2)
    slots = threading.BoundedSemaphore(2)
    active = peak = closed = 0
    diagnostics = []

    def reset():
        assert slots.acquire(timeout=1), "reset starved step/close threads"

    async def episode(*args, diagnostic, **kwargs):
        nonlocal active, peak, closed
        await asyncio.to_thread(reset)
        active += 1
        peak = max(peak, active)
        diagnostics.append(diagnostic)
        try:
            await asyncio.sleep(0.001)  # Model generation leaves threads free.
            await asyncio.to_thread(lambda: None)  # Environment step.
        finally:
            await asyncio.to_thread(slots.release)
            active -= 1
            closed += 1

    monkeypatch.setattr(slime_runtime, "_run_episode_impl", episode)

    async def run():
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=2)
        )
        await asyncio.wait_for(
            asyncio.gather(
                *(_episode(args, diagnostic=i % 2 == 0) for i in range(128))
            ),
            timeout=5,
        )

    asyncio.run(run())
    assert peak == 2
    assert active == 0
    assert closed == 128
    assert diagnostics.count(True) == diagnostics.count(False) == 64


def test_episode_admission_releases_after_error_and_cancellation(monkeypatch):
    args = SimpleNamespace(shopsim_episode_concurrency=1)

    async def run():
        started = asyncio.Event()
        allow_close = asyncio.Event()
        closing = asyncio.Event()
        calls = 0

        async def episode(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("episode failed")
            if calls == 2:
                started.set()
                try:
                    await asyncio.Future()
                finally:
                    closing.set()
                    await allow_close.wait()
            return "completed"

        monkeypatch.setattr(slime_runtime, "_run_episode_impl", episode)
        with pytest.raises(RuntimeError, match="episode failed"):
            await _episode(args)

        running = asyncio.create_task(_episode(args))
        await started.wait()
        waiting = asyncio.create_task(_episode(args, diagnostic=True))
        await asyncio.sleep(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        running.cancel()
        await closing.wait()
        following = asyncio.create_task(_episode(args, diagnostic=True))
        await asyncio.sleep(0)
        assert calls == 2  # Cancellation must finish cleanup before admission.
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert await following == "completed"

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_episode_admission_is_recreated_for_new_event_loop(monkeypatch):
    args = SimpleNamespace(shopsim_episode_concurrency=1)

    async def episode(*args, **kwargs):
        await asyncio.sleep(0)

    monkeypatch.setattr(slime_runtime, "_run_episode_impl", episode)

    async def run():
        await asyncio.gather(_episode(args), _episode(args))

    asyncio.run(run())
    asyncio.run(run())


def test_group_retry_reenters_admission_without_deadlock(tmp_path, monkeypatch):
    args = SimpleNamespace(
        shopsim_episode_concurrency=1,
        n_samples_per_prompt=8,
        shopsim_curriculum_path=str(tmp_path / "curriculum.json"),
        shopsim_training_output_dir=str(tmp_path),
    )
    assignment = SkillAssignment(
        state_id="test-state", group_key="0", mode="skill_free", skills=()
    )
    curriculum = SimpleNamespace(
        state_id=assignment.state_id,
        round_id="test-round",
        assign=lambda _: assignment,
        identity=lambda: {"state_id": assignment.state_id},
        full_skills=lambda: (),
    )
    calls = []

    async def episode(args, sample, sampling, *, diagnostic, **kwargs):
        calls.append(diagnostic)
        await asyncio.sleep(0)
        return slime_runtime.EpisodeRollout(
            trace={"episode_id": str(sample.index)},
            samples=[SimpleNamespace(metadata={})],
            reward={"reward": float(diagnostic), "r_success": float(diagnostic)},
            scored=True,
        )

    monkeypatch.setattr(slime_runtime, "_curriculum", lambda *args: curriculum)
    monkeypatch.setattr(slime_runtime, "_run_episode_impl", episode)

    async def run():
        await asyncio.wait_for(
            asyncio.gather(
                *(slime_runtime.generate(args, SimpleNamespace(index=i, group_index=0, prompt="7"), {})
                  for i in range(8))
            ),
            timeout=5,
        )

    asyncio.run(run())
    assert calls == [False] * 8 + [True]
    triage = json.loads(
        (tmp_path / "test-round/group-000000000/triage.json").read_text(encoding="utf-8")
    )
    assert triage["classification"] == "model_internalization_deficit"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", None])
def test_episode_admission_rejects_invalid_limit(limit):
    async def run():
        with pytest.raises(ValueError, match="must be a positive integer"):
            await _episode(SimpleNamespace(shopsim_episode_concurrency=limit))

    asyncio.run(run())
