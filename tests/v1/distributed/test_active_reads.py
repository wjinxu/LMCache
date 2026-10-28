# SPDX-License-Identifier: Apache-2.0
"""Active reads retain the exact allocation until asynchronous use ends."""

# Standard
from collections.abc import Iterator
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ActiveRead
from lmcache.v1.distributed.l1_manager import L1Manager
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
KEY = ObjectKey(ObjectKey.IntHash2Bytes(1), "active-read", 0)
OTHER = ObjectKey(ObjectKey.IntHash2Bytes(2), "active-read", 0)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> Iterator[L1Manager]:
    """Use real L1 locks and allocation with a single unpinned CPU slot."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    l1 = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False, shm_name=""),
            read_ttl_seconds=1,
        )
    )
    yield l1
    l1.close()


def admit(manager: L1Manager, key: ObjectKey = KEY) -> None:
    """Admit one object through the public write lifecycle."""
    assert manager.reserve_write([key], [False], LAYOUT)[key][0] == L1Error.SUCCESS
    assert manager.finish_write([key])[key] == L1Error.SUCCESS


def acquire(manager: L1Manager) -> L1ActiveRead:
    """Convert the caller's prefetch reservation to an active read."""
    error, read = manager.acquire_read([KEY])[KEY]
    assert error == L1Error.SUCCESS and read is not None
    return read


def test_active_read_survives_ttl_and_blocks_reuse(manager: L1Manager) -> None:
    """Expiry cannot expose an acquired buffer to eviction or reuse."""
    admit(manager)
    manager.reserve_read([KEY])
    read = acquire(manager)
    time.sleep(1.1)
    assert not manager.is_key_evictable(KEY)
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.clear()
    assert read.memory_obj.is_valid()
    assert manager.report_status()["read_locked_count"] == 1
    assert (
        manager.reserve_write([OTHER], [False], LAYOUT)[OTHER][0]
        == L1Error.OUT_OF_MEMORY
    )
    assert manager.finish_active_reads([read.handle]) == [KEY]
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS
    admit(manager, OTHER)


@pytest.mark.parametrize("temporary", [False, True])
def test_last_active_reader_controls_reclamation(
    manager: L1Manager, temporary: bool
) -> None:
    """One reader finishing must not recycle another reader's allocation."""
    manager.reserve_write([KEY], [temporary], LAYOUT)
    manager.finish_write_and_reserve_read([KEY], read_locks=2)
    first, second = acquire(manager), acquire(manager)
    manager.finish_active_reads([first.handle, first.handle])
    assert second.memory_obj.is_valid()
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_active_reads([second.handle])
    assert second.memory_obj.is_valid() is not temporary
    if not temporary:
        assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_completion_does_not_release_new_reservation(manager: L1Manager) -> None:
    """An active completion cannot decrement a newer lookup's TTL count."""
    admit(manager)
    manager.reserve_read([KEY])
    old = acquire(manager)
    time.sleep(1.1)
    manager.reserve_read([KEY])
    manager.finish_active_reads([old.handle])
    assert manager.unsafe_read([KEY])[KEY][0] == L1Error.SUCCESS
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_read([KEY])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_old_handle_cannot_release_replacement(manager: L1Manager) -> None:
    """Completion replay is harmless even after the same key is reused."""
    admit(manager)
    manager.reserve_read([KEY])
    old = acquire(manager)
    manager.finish_active_reads([old.handle])
    manager.delete([KEY])
    admit(manager)
    manager.reserve_read([KEY])
    current = acquire(manager)
    assert current.handle != old.handle
    assert manager.finish_active_reads([old.handle]) == []
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_active_reads([current.handle])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_expired_or_absent_reservation_is_not_acquired(manager: L1Manager) -> None:
    """Active protection cannot resurrect a missing or expired reservation."""
    assert manager.acquire_read([KEY])[KEY] == (L1Error.KEY_NOT_EXIST, None)
    admit(manager)
    assert manager.acquire_read([KEY])[KEY] == (L1Error.KEY_NOT_READABLE, None)
    manager.reserve_read([KEY])
    time.sleep(1.1)
    assert manager.acquire_read([KEY])[KEY] == (L1Error.KEY_NOT_READABLE, None)
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS


def test_forced_delete_detaches_handle_from_replacement(manager: L1Manager) -> None:
    """Destructive administrative deletion cannot redirect a late callback."""
    admit(manager)
    manager.reserve_read([KEY])
    old = acquire(manager)
    manager.delete([KEY], force=True)
    admit(manager)
    manager.reserve_read([KEY])
    current = acquire(manager)
    assert manager.finish_active_reads([old.handle]) == []
    assert current.memory_obj.is_valid()
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_active_reads([current.handle])


def test_legacy_completion_keeps_temporary_active_read(manager: L1Manager) -> None:
    """A separate TTL reader cannot free a temporary transfer's buffer."""
    manager.reserve_write([KEY], [True], LAYOUT)
    manager.finish_write_and_reserve_read([KEY], read_locks=2)
    read = acquire(manager)
    assert manager.finish_read([KEY])[KEY] == L1Error.SUCCESS
    assert read.memory_obj.is_valid()
    assert manager.delete([KEY])[KEY] == L1Error.KEY_IS_LOCKED
    manager.finish_active_reads([read.handle])
    assert not read.memory_obj.is_valid()


def test_partial_acquisition_and_duplicate_keys(manager: L1Manager) -> None:
    """One reservation is consumed per distinct key and misses stay misses."""
    admit(manager)
    manager.reserve_read([KEY])
    result = manager.acquire_read([KEY, OTHER, KEY])
    error, read = result[KEY]
    assert error == L1Error.SUCCESS and read is not None
    assert result[OTHER] == (L1Error.KEY_NOT_EXIST, None)
    assert manager.acquire_read([KEY])[KEY] == (L1Error.KEY_NOT_READABLE, None)
    manager.finish_active_reads([read.handle])
    assert manager.delete([KEY])[KEY] == L1Error.SUCCESS
