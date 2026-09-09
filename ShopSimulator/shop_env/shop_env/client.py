"""Stateful HTTP client suitable for rollout workers and evaluators."""

from __future__ import annotations

from typing import Any

import requests


class ShopSimulatorError(RuntimeError):
    pass


class ShopSimulatorClient:
    """One client instance owns at most one environment lease."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5700",
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ):
        self.api_url = f"{base_url.rstrip('/')}/api/shop_agent"
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        self.env_idx: int | None = None
        self.lease_id: str | None = None

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = (
            (self.timeout, None)
            if payload.get("action") == "reset"
            else self.timeout
        )
        response = self.session.post(
            self.api_url, json=payload, timeout=timeout
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise ShopSimulatorError(
                f"ShopSimulator returned non-JSON HTTP {response.status_code}"
            ) from exc
        result = body.get("result")
        if not isinstance(result, dict):
            raise ShopSimulatorError("ShopSimulator response has no result object")
        if not response.ok or result.get("error"):
            raise ShopSimulatorError(
                f"ShopSimulator HTTP {response.status_code}: "
                f"{result.get('error', response.reason)}"
            )
        if payload.get("action") in {"reset", "interact", "terminate"}:
            if not isinstance(result.get("observation_state"), dict):
                raise ShopSimulatorError(
                    "ShopSimulator response has no canonical observation_state"
                )
            if not isinstance(result.get("observation"), str):
                raise ShopSimulatorError(
                    "ShopSimulator response has no rendered observation"
                )
            if "instruction" in result:
                raise ShopSimulatorError(
                    "ShopSimulator returned the retired instruction page field"
                )
            if payload.get("action") == "reset" and not isinstance(
                result.get("task_instruction"), str
            ):
                raise ShopSimulatorError(
                    "ShopSimulator reset has no task_instruction"
                )
        return result

    def reset(
        self,
        task_id: int,
        *,
        persona: bool = False,
        include_private_goal: bool = False,
        interaction_mode: str = "single_turn",
    ) -> dict[str, Any]:
        if self.env_idx is not None:
            raise ShopSimulatorError("client already owns a lease; release it first")
        result = self._post(
            {
                "action": "reset",
                "idx": task_id,
                "if_persona": bool(persona),
                "include_private_goal": bool(include_private_goal),
                "interaction_mode": interaction_mode,
            }
        )
        self.env_idx = int(result["env_idx"])
        self.lease_id = str(result["lease_id"])
        return result

    def step(self, action: str) -> dict[str, Any]:
        if self.env_idx is None or self.lease_id is None:
            raise ShopSimulatorError("reset must be called before step")
        return self._post(
            {
                "action": "interact",
                "env_idx": self.env_idx,
                "lease_id": self.lease_id,
                "response": action,
            }
        )

    def terminate(self, reason: str) -> dict[str, Any]:
        if self.env_idx is None or self.lease_id is None:
            raise ShopSimulatorError("reset must be called before termination")
        if reason not in {"action_limit", "generation_length"}:
            raise ValueError(f"unsupported termination reason: {reason}")
        return self._post(
            {
                "action": "terminate",
                "env_idx": self.env_idx,
                "lease_id": self.lease_id,
                "termination_reason": reason,
            }
        )

    def release(self) -> dict[str, Any] | None:
        if self.env_idx is None or self.lease_id is None:
            return None
        env_idx, lease_id = self.env_idx, self.lease_id
        try:
            return self._post(
                {
                    "action": "release_one",
                    "env_idx": env_idx,
                    "lease_id": lease_id,
                }
            )
        finally:
            self.env_idx = None
            self.lease_id = None

    def close(self) -> None:
        try:
            self.release()
        finally:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
