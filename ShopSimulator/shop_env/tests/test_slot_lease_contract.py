import unittest
import threading

from shop_env.slot_lease_pool import LeaseMismatchError, SlotLeasePool


class SlotLeasePoolTest(unittest.TestCase):
    def test_terminal_owner_keeps_slot_until_explicit_release(self):
        pool = SlotLeasePool(1)
        slot = pool.acquire()
        self.assertEqual(slot, 0)
        self.assertIsNone(pool.acquire())
        self.assertTrue(pool.release(slot))
        self.assertEqual(pool.acquire(), 0)

    def test_stale_duplicate_release_is_visible(self):
        pool = SlotLeasePool(1)
        slot = pool.acquire()
        self.assertTrue(pool.release(slot))
        self.assertFalse(pool.release(slot))

    def test_reset_recovers_all_slots(self):
        pool = SlotLeasePool(3)
        pool.acquire()
        pool.acquire()
        pool.reset(3)
        self.assertEqual(pool.free_slots(), frozenset({0, 1, 2}))

    def test_stale_owner_cannot_release_reassigned_slot(self):
        pool = SlotLeasePool(1)
        slot = pool.acquire()
        first_lease = pool.lease_id(slot)
        pool.release(slot, first_lease)

        reassigned = pool.acquire()
        second_lease = pool.lease_id(reassigned)
        self.assertNotEqual(first_lease, second_lease)
        with self.assertRaises(LeaseMismatchError):
            pool.release(reassigned, first_lease)
        self.assertTrue(pool.validate(reassigned, second_lease))

    def test_blocking_acquire_waits_until_a_slot_is_released(self):
        pool = SlotLeasePool(1)
        active_slot = pool.acquire()
        waiter_started = threading.Event()
        waiter_finished = threading.Event()
        acquired = []

        def wait_for_slot():
            waiter_started.set()
            acquired.append(pool.acquire(block=True))
            waiter_finished.set()

        waiter = threading.Thread(target=wait_for_slot)
        waiter.start()
        self.assertTrue(waiter_started.wait(timeout=1))
        self.assertFalse(waiter_finished.wait(timeout=0.05))

        pool.release(active_slot)
        self.assertTrue(waiter_finished.wait(timeout=1))
        waiter.join(timeout=1)
        self.assertEqual(acquired, [active_slot])
        pool.release(acquired[0])


if __name__ == "__main__":
    unittest.main()
