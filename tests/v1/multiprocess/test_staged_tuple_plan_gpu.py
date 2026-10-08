# SPDX-License-Identifier: Apache-2.0
"""Staged tuple plans preserve bytes, skips and staging-buffer reuse."""

# Standard
from dataclasses import dataclass
from typing import Iterator
import random

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops, torch_device_type
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.object_group_transfer import (
    downsample_and_stage_block_ids,
    transfer_kv_per_object_group,
)
from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext
import lmcache.lmcache_native as native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        torch_device_type != "cuda"
        or not hasattr(device_ops.BatchStep, "from_staging_tuples"),
        reason="Requires the CUDA staging-tuple batch factory",
    ),
]


@dataclass
class LocalTensor:
    """Expose a same-process tensor through the cache context's IPC interface."""

    tensor: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        """Return the locally owned tensor."""
        return self.tensor

    def close(self) -> None:
        """Keep the caller's tensor alive."""


@pytest.fixture
def allocator() -> Iterator[LazyMemoryAllocator]:
    alloc = LazyMemoryAllocator(init_size=1 << 20, final_size=1 << 20)
    yield alloc
    alloc.close()


@pytest.mark.parametrize(
    "direction", [native.TransferDirection.H2D, native.TransferDirection.D2H]
)
@pytest.mark.parametrize("skip", [0, 8, 40, 160])
@pytest.mark.parametrize("window", [-1, 16])
def test_tuple_plan_matches_legacy(
    direction: native.TransferDirection,
    skip: int,
    window: int,
    allocator: LazyMemoryAllocator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two groups, five chunks, tail batches and windows match the old backend."""
    torch.manual_seed(7)
    tensors = [
        torch.randint(0, 100, (2, 64, 8, nh, hs), dtype=torch.int16, device="cuda")
        for nh, hs in [(2, 16), (1, 32)]
    ]
    ctx = GPUCacheContext(
        [LocalTensor(t) for t in tensors],  # type: ignore[misc]
        lmcache_tokens_per_chunk=32,
        engine_group_infos=[
            EngineGroupInfo(0, (i,), sw_size_tokens=window) for i in range(2)
        ],
        separate_object_groups=window > 0,
    )
    outputs: list[list[torch.Tensor]] = []
    rng = random.Random(7)
    host_ids = [rng.sample(range(1, 64), 20) for _ in tensors]
    objects: list[list[MemoryObj]] = []
    before = [t.clone() for t in tensors]
    try:
        for group_idx in range(len(ctx.kv_layer_groups_manager.object_groups)):
            size = ctx.get_temp_object_group_buffer(0, group_idx).nbytes
            group_objects: list[MemoryObj] = []
            for _ in range(5):
                obj = allocator.allocate(torch.Size([size]), torch.uint8)
                assert obj is not None and obj.raw_tensor is not None
                obj.raw_tensor.copy_(torch.randint(0, 255, (size,), dtype=torch.uint8))
                group_objects.append(obj)
            objects.append(group_objects)
        initial_objects: list[torch.Tensor] = []
        for group in objects:
            for obj in group:
                assert obj.raw_tensor is not None
                initial_objects.append(obj.raw_tensor.clone())
        for legacy in [False, True]:
            for tensor, original in zip(tensors, before, strict=True):
                tensor.copy_(original)
            for obj, original in zip(
                (obj for group in objects for obj in group),
                initial_objects,
                strict=True,
            ):
                assert obj.raw_tensor is not None
                obj.raw_tensor.copy_(original)
            with monkeypatch.context() as patch:
                if legacy:
                    patch.delattr(device_ops.BatchStep, "from_staging_tuples")
                with torch.cuda.stream(ctx.stream):
                    # D2H leaves skipped prefix bytes in reused staging slots.
                    for slot in range(ctx.max_batch_size):
                        for group_idx in range(len(objects)):
                            ctx.get_temp_object_group_buffer(slot, group_idx).zero_()
                    ids = downsample_and_stage_block_ids(
                        ctx, [g.copy() for g in host_ids]
                    )
                    for group_idx, group_objects in enumerate(objects):
                        transfer_kv_per_object_group(
                            ctx,
                            ids,
                            group_objects,
                            group_idx,
                            4,
                            skip,
                            direction,
                            transfer_key="tuple-plan-test",
                        )
                ctx.stream.synchronize()
            outputs.append(
                [t.clone() for t in tensors]
                if direction == native.TransferDirection.H2D
                else [obj.raw_tensor.clone() for group in objects for obj in group]  # type: ignore[union-attr]
            )
        assert all(torch.equal(a, b) for a, b in zip(*outputs, strict=True))
        original_outputs = (
            before if direction == native.TransferDirection.H2D else initial_objects
        )
        changed = any(
            not torch.equal(a, b)
            for a, b in zip(outputs[0], original_outputs, strict=True)
        )
        assert changed == (skip < 160)
    finally:
        for group in objects:
            for obj in group:
                allocator.free(obj)
        ctx.close()


@pytest.mark.parametrize(
    "copies",
    [[(1, 1, 8)], [(1, 1, -1, 0)], [(1, 1, 8, 1 << 80)]],
)
def test_malformed_batch_rejected(copies: list[tuple[int, ...]]) -> None:
    """Fixed tuple arity and C++ integer widths are checked at construction."""
    factory = getattr(device_ops.BatchStep, "from_staging_tuples", None)
    assert factory is not None
    with pytest.raises(TypeError):
        factory(copies, [])
