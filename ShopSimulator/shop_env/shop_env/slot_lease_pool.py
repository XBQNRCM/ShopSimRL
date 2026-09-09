"""Thread-safe explicit lease tracking for ShopSimulator worker slots."""

from __future__ import annotations

import threading
import uuid


class LeaseMismatchError(RuntimeError):
    """Raised when a stale caller tries to operate on a reassigned slot."""


class SlotLeasePool:
    """A slot remains leased until its owner explicitly releases it."""

    def __init__(self, size: int):
        self._condition = threading.Condition()
        self._size = 0
        self._free: set[int] = set()
        self._lease_ids: dict[int, str] = {}
        self.reset(size)

    def reset(self, size: int) -> None:
        size = int(size)
        if size < 0:
            raise ValueError("slot pool size must be non-negative")
        with self._condition:
            self._size = size
            self._free = set(range(size))
            self._lease_ids = {}
            self._condition.notify_all()

    def acquire(self, *, block: bool = False) -> int | None:
        with self._condition:
            while not self._free:
                if not block:
                    return None
                self._condition.wait()
            slot = self._free.pop()
            self._lease_ids[slot] = uuid.uuid4().hex
            return slot

    def lease_id(self, slot: int) -> str | None:
        with self._condition:
            return self._lease_ids.get(int(slot))

    def validate(self, slot: int, lease_id: object) -> bool:
        with self._condition:
            return self._lease_ids.get(int(slot)) == str(lease_id)

    def release(self, slot: int, lease_id: object = None) -> bool:
        slot = int(slot)
        with self._condition:
            if slot < 0 or slot >= self._size:
                raise ValueError(f"invalid environment index: {slot}")
            was_leased = slot not in self._free
            if was_leased and lease_id is not None:
                active_lease = self._lease_ids.get(slot)
                if active_lease != str(lease_id):
                    raise LeaseMismatchError(
                        f"stale or invalid lease for environment {slot}"
                    )
            self._free.add(slot)
            self._lease_ids.pop(slot, None)
            if was_leased:
                self._condition.notify()
            return was_leased

    def free_slots(self) -> frozenset[int]:
        with self._condition:
            return frozenset(self._free)
