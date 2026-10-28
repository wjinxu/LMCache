# Active reads for MP GPU retrieve

The ordinary `LMCacheDrivenTransferModule.retrieve` path queues asynchronous
host-to-device work. A Python reference to a `MemoryObj` does not keep its pool
allocation resident: after its read TTL expires, normal eviction can free the
slot and another write can reuse the address before the device reads it.

## Two lifetimes

Prefetch reservations remain TTL-based so abandoned requests do not retain L1
capacity forever. Immediately before GPU transfer, `L1Manager.acquire_read`
converts one live reservation per distinct key into an active read under the
same manager lock used by eviction. It validates residency and the TTL,
increments the resident entry's active count, consumes one TTL reservation,
and returns `L1ActiveRead(handle, memory_obj)`.

```text
prefetch reserves a TTL read
          |
          +-- expires before acquisition --> miss; no GPU copy
          |
          +-- acquire_read (under L1 lock)
                      |
          consume one TTL reservation; create unique handle
                      |
          enqueue H2D; active count prevents normal reclamation
                      |
          stream callback --> finish_active_reads(handles)
```

An active read has no timeout. Normal `delete`, `clear`, temporary-object
cleanup and eviction eligibility respect the active count. The final active
completion frees a temporary object only if no live TTL reservation remains.
Status reporting counts both forms of protection as read-locked objects.

## Completion and ownership

Each L1 allocates monotonically increasing, manager-local integer handles.
Its handle table retains the exact resident entry, not just its key. Completion
removes that acquisition once; replaying a handle is harmless and cannot
decrement a newer reservation. A late completion after forced deletion cannot
affect a replacement object with the same key.

`StorageManager.acquire_read(keys, l1_owners)` uses the exact owners retained
by prefetch and returns successful keys, objects in request order, and
`L1ReadCompletion = list[tuple[int, list[int]]]`. Each tuple contains an L1
manager ID and its acquisition handles. This payload is MessagePack-compatible;
no `MemoryObj` or pointer crosses the native callback boundary. Multi-L1 reads
require explicit owners, as with the existing serving read APIs.

Storage tracing records successful completion as the existing key-based
`finish_read_prefetched(keys, read_locks=1)` operation. Replay retains TTL
reservations rather than capturing process-local handles or object pointers;
this preserves its release stream without adding a new trace schema. Replay
does not reproduce the active allocation lifetime during asynchronous device
work.

Acquisition may partially succeed. The caller must release every returned
handle even if another key is unavailable. If a later L1 raises during
acquisition, StorageManager releases the acquisitions it already received.

## MP transfer integration

Retrieve acquires every retained group before queuing any object transfers.
It preserves sliding-window suffix selection and skips connector-private aux
groups. Missing acquisitions fail retrieve before copying any object. A transfer
exception may leave earlier device work queued; the `finally` path schedules
`finish_active_reads` after that work on the same stream, including handles for
groups that the failed loop never visited. The hot path adds no stream
synchronization or renewal thread.

The cost is one handle-table entry per active object and one extra manager
critical section per acquisition batch. Holding all groups before copying can
increase the duration of memory pressure. A lost completion callback retains
the active allocations until destructive cleanup or shutdown; a timeout cannot
safely reclaim memory while a device may still access it. Device/dispatcher
failure recovery is outside this change.

## Boundaries

- `force=True` deletion/clear and `close()` retain their destructive contract;
  callers must quiesce device users first. Object identity prevents a late
  callback from releasing a replacement, but cannot make forced buffer reuse
  safe for an in-flight transfer.
- Acquisition verifies that a reservation is live, but anonymous TTL counts
  cannot identify which lookup owns a renewed reservation. The stale queued
  lookup check in PR #5797 remains complementary.
- CacheBlend, P2P, SHM, inline reads, L2 store and serde consumers keep their
  existing read protocols. This change does not establish a global guarantee
  for every asynchronous consumer.
- The implementation lives in `lmcache/v1/distributed/` and
  `lmcache/v1/multiprocess/`, which are MP components. It does not change the
  in-process cache engine.

## Regression coverage

- `test_active_reads.py`: a single-slot allocator cannot be reused after TTL
  while acquired; concurrent readers, temporary cleanup, newer reservations,
  repeated completion, replacement identity and partial acquisition.
- `test_multi_l1.py`: exact owner routing, MessagePack round trip, partial
  success cleanup, rollback after a later owner fails and completion replay.
- `test_lmcache_driven_transfer_skip.py`: window and aux selection, releasing
  all retained acquisitions after a partly enqueued copy raises, and skipping
  transfers while releasing successful handles on a partial miss.

See [L1ManagerInterface](l1_manager_interface.md) and
[StorageManager](storage_manager.md) for the surrounding contracts.
