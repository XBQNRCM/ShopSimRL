"""Canonical ShopSimulator HTTP boundary used by rollout workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


class ShopEnvironment(Protocol):
    def identity(self) -> dict[str, Any]: ...

    def reset(self, task_id: int) -> dict[str, Any]: ...

    def step(self, action: str) -> dict[str, Any]: ...

    def terminate(self, reason: str) -> dict[str, Any]: ...

    def close(self) -> None: ...


class ShopSimulatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShopSimulatorConfig:
    base_url: str = "http://127.0.0.1:5700"
    persona: bool = False
    interaction_mode: str = "single_turn"
    timeout: float = 30.0
    trust_env: bool = False

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.interaction_mode.strip():
            raise ValueError("base_url and interaction_mode must be non-empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def identity(self) -> dict[str, Any]:
        return {
            "adapter": "shopsimulator_http",
            "base_url": self.base_url.rstrip("/"),
            "persona": self.persona,
            "interaction_mode": self.interaction_mode,
        }


class ShopSimulatorHTTPEnvironment:
    """One instance owns at most one server-side environment lease."""

    def __init__(self, config: ShopSimulatorConfig):
        self.config = config
        self.api_url = f"{config.base_url.rstrip('/')}/api/shop_agent"
        self.session = requests.Session()
        self.session.trust_env = config.trust_env
        self.env_idx: int | None = None
        self.lease_id: str | None = None
        self.environment_version: str | None = None
        self.observation_version: str | None = None

    def identity(self) -> dict[str, Any]:
        identity = self.config.identity()
        identity.update(
            {
                "environment_version": self.environment_version,
                "observation_version": self.observation_version,
            }
        )
        return identity

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = (
            (self.config.timeout, None)
            if payload.get("action") == "reset"
            else self.config.timeout
        )
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=timeout,
            )
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ShopSimulatorError(f"ShopSimulator request failed: {exc}") from exc
        result = body.get("result")
        if not isinstance(result, dict):
            raise ShopSimulatorError(
                f"ShopSimulator HTTP {response.status_code} has no result object"
            )
        if not response.ok or result.get("error"):
            raise ShopSimulatorError(
                f"ShopSimulator HTTP {response.status_code}: "
                f"{result.get('error', response.reason)}"
            )
        if payload["action"] in {"reset", "interact", "terminate"}:
            self._validate_observation(result)
        return result

    @staticmethod
    def _validate_observation(result: dict[str, Any]) -> None:
        if not isinstance(result.get("observation"), str):
            raise ShopSimulatorError("response has no rendered observation")
        state = result.get("observation_state")
        if not isinstance(state, dict):
            raise ShopSimulatorError("response has no canonical observation_state")
        if not isinstance(state.get("observation_version"), str):
            raise ShopSimulatorError("observation_state has no version")

    def reset(self, task_id: int) -> dict[str, Any]:
        if self.env_idx is not None:
            raise ShopSimulatorError("environment already owns a lease")
        result = self._post(
            {
                "action": "reset",
                "idx": task_id,
                "if_persona": self.config.persona,
                "include_private_goal": False,
                "interaction_mode": self.config.interaction_mode,
            }
        )
        if not isinstance(result.get("task_instruction"), str):
            raise ShopSimulatorError("reset has no task_instruction")
        self.env_idx = int(result["env_idx"])
        self.lease_id = str(result["lease_id"])
        self.environment_version = result.get("environment_version")
        self.observation_version = result["observation_state"].get(
            "observation_version"
        )
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
        result = self._post(
            {
                "action": "terminate",
                "env_idx": self.env_idx,
                "lease_id": self.lease_id,
                "termination_reason": reason,
            }
        )
        if result.get("termination_reason") != reason:
            raise ShopSimulatorError(
                "ShopSimulator returned a mismatched termination reason; "
                "restart the environment server with the current code"
            )
        return result

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
