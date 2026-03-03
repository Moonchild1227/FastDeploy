# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Head-wise KV Cache Allocator for True cache_id-based Architecture

This allocator manages cache IDs that are completely free (not encoded with head information).
Each head can independently allocate and recycle cache IDs from a shared free list.
"""

import heapq
import time
from typing import Dict, List

from fastdeploy.utils import get_logger

logger = get_logger("headwise_allocator", "cache_manager.log")


class HeadWiseCacheAllocator:
    """
    Head-wise cache allocator with true cache_id-based architecture.

    Key differences from view-based approach:
    - cache_id is completely free (no block_id * kv_heads + head_id encoding)
    - Physical layout is [cache_id, block_size, head_dim] (not a view)
    - Each head independently allocates from shared free_list
    - Supports unequal head lengths (head-wise SWA)

    Example:
        >>> allocator = HeadWiseCacheAllocator(total_cache_ids=1000, kv_num_heads=4)
        >>> # Equal allocation: all heads get 16 blocks
        >>> cache_ids = allocator.allocate_per_head([16, 16, 16, 16])
        >>> # Unequal allocation: heads get different numbers
        >>> cache_ids = allocator.allocate_per_head([16, 12, 16, 12])
        >>> # Recycle
        >>> allocator.recycle(cache_ids)
    """

    def __init__(self, total_cache_ids: int, kv_num_heads: int):
        """
        Initialize the head-wise cache allocator.

        Args:
            total_cache_ids: Total number of cache IDs available
                           (typically num_gpu_blocks * kv_num_heads)
            kv_num_heads: Number of KV heads (for GQA/MQA)
        """
        self.total_cache_ids = total_cache_ids
        self.kv_num_heads = kv_num_heads

        # Free list maintains available cache IDs (min-heap for efficient allocation)
        # Initialize in reverse order so we allocate from 0, 1, 2, ... sequentially
        self.free_list = list(range(total_cache_ids - 1, -1, -1))
        heapq.heapify(self.free_list)

        # Statistics tracking
        self.allocation_stats = {
            "total_allocations": 0,
            "total_recyclations": 0,
            "per_head_allocations": [0] * kv_num_heads,
            "current_usage": 0,
        }

        # Cache ID to position mapping (for debugging/validation)
        self.cache_id_metadata: Dict[int, dict] = {}

        logger.info(
            f"[HEAD_WISE] Initialized HeadWiseCacheAllocator: "
            f"total_cache_ids={total_cache_ids}, kv_num_heads={kv_num_heads}, "
            f"available={len(self.free_list)}"
        )

    def allocate_per_head(self, num_blocks_per_head: List[int]) -> List[List[int]]:
        """
        Allocate cache IDs for each head independently.

        Each head gets its own cache IDs from the shared free list.
        Different heads can have different numbers of blocks (head-wise SWA).

        Args:
            num_blocks_per_head: Number of blocks to allocate for each head
                                 [head_0_blocks, head_1_blocks, ...]

        Returns:
            2D list of cache IDs: [[head_0_cache_ids], [head_1_cache_ids], ...]

        Raises:
            RuntimeError: If insufficient cache IDs available
        """
        if len(num_blocks_per_head) != self.kv_num_heads:
            raise ValueError(
                f"[HEAD_WISE] Expected {self.kv_num_heads} head counts, " f"got {len(num_blocks_per_head)}"
            )

        required_cache_ids = sum(num_blocks_per_head)
        available_cache_ids = len(self.free_list)

        if required_cache_ids > available_cache_ids:
            raise RuntimeError(
                f"[HEAD_WISE] Insufficient cache: need {required_cache_ids}, "
                f"but only {available_cache_ids} available"
            )

        cache_ids_2d = []
        for head_id, num_blocks in enumerate(num_blocks_per_head):
            head_cache_ids = []
            for _ in range(num_blocks):
                if not self.free_list:
                    raise RuntimeError("[HEAD_WISE] Free list exhausted during allocation")

                cache_id = heapq.heappop(self.free_list)

                # Record metadata for debugging
                self.cache_id_metadata[cache_id] = {
                    "head_id": head_id,
                    "allocated_at": time.time(),
                }

                head_cache_ids.append(cache_id)
                self.allocation_stats["per_head_allocations"][head_id] += 1

            cache_ids_2d.append(head_cache_ids)

        self.allocation_stats["total_allocations"] += required_cache_ids
        self.allocation_stats["current_usage"] += required_cache_ids

        logger.debug(
            f"[HEAD_WISE] Allocated {num_blocks_per_head} blocks per head, "
            f"total {required_cache_ids} cache_ids, "
            f"remaining: {len(self.free_list)}"
        )

        return cache_ids_2d

    def recycle(self, cache_ids_2d: List[List[int]]):
        """
        Recycle cache IDs back to the free list.

        Args:
            cache_ids_2d: 2D list of cache IDs to recycle
                        [[head_0_cache_ids], [head_1_cache_ids], ...]
        """
        recycled_count = 0

        for head_id, head_cache_ids in enumerate(cache_ids_2d):
            for cache_id in head_cache_ids:
                # Remove metadata
                if cache_id in self.cache_id_metadata:
                    del self.cache_id_metadata[cache_id]

                # Return to free list
                heapq.heappush(self.free_list, cache_id)
                recycled_count += 1

                # Update per-head stats (optional: track recyclations per head)
                self.allocation_stats["per_head_allocations"][head_id] -= 1

        self.allocation_stats["total_recyclations"] += recycled_count
        self.allocation_stats["current_usage"] -= recycled_count

        logger.debug(f"[HEAD_WISE] Recycled {recycled_count} cache_ids, " f"remaining: {len(self.free_list)}")

    def extend_allocation(
        self, existing_cache_ids: List[List[int]], additional_blocks_per_head: List[int]
    ) -> List[List[int]]:
        """
        Extend existing allocation with additional blocks for each head.

        Args:
            existing_cache_ids: Current 2D cache IDs
            additional_blocks_per_head: Additional blocks for each head
                                        [head_0_add, head_1_add, ...]

        Returns:
            Extended 2D cache IDs (modified in-place and returned)

        Raises:
            RuntimeError: If insufficient cache IDs for extension
        """
        if len(additional_blocks_per_head) != self.kv_num_heads:
            raise ValueError(
                f"[HEAD_WISE] Expected {self.kv_num_heads} head counts, " f"got {len(additional_blocks_per_head)}"
            )

        required = sum(additional_blocks_per_head)
        if required > len(self.free_list):
            raise RuntimeError(
                f"[HEAD_WISE] Insufficient cache for extension: need {required}, " f"available {len(self.free_list)}"
            )

        for head_id, additional_blocks in enumerate(additional_blocks_per_head):
            for _ in range(additional_blocks):
                cache_id = heapq.heappop(self.free_list)

                self.cache_id_metadata[cache_id] = {
                    "head_id": head_id,
                    "allocated_at": time.time(),
                }

                existing_cache_ids[head_id].append(cache_id)
                self.allocation_stats["per_head_allocations"][head_id] += 1

        self.allocation_stats["total_allocations"] += required
        self.allocation_stats["current_usage"] += required

        logger.info(
            f"[HEAD_WISE] Extended allocation by {additional_blocks_per_head} blocks, " f"total {required} cache_ids"
        )

        return existing_cache_ids

    def calculate_fragmentation(self) -> float:
        """
        Calculate memory fragmentation ratio.

        Fragmentation = 1 - (max_consecutive_segment / total_free)

        Returns:
            Fragmentation ratio between 0.0 (no fragmentation) and 1.0 (fully fragmented)
        """
        if not self.free_list:
            return 0.0

        sorted_ids = sorted(self.free_list)
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(sorted_ids)):
            if sorted_ids[i] == sorted_ids[i - 1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        fragmentation = 1.0 - (max_consecutive / len(self.free_list))
        return fragmentation

    def get_stats(self) -> dict:
        """
        Get allocator statistics.

        Returns:
            Dictionary with current statistics
        """
        fragmentation = self.calculate_fragmentation()

        return {
            "total_cache_ids": self.total_cache_ids,
            "available_cache_ids": len(self.free_list),
            "current_usage": self.allocation_stats["current_usage"],
            "total_allocations": self.allocation_stats["total_allocations"],
            "total_recyclations": self.allocation_stats["total_recyclations"],
            "fragmentation": fragmentation,
            "per_head_allocations": self.allocation_stats["per_head_allocations"].copy(),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"HeadWiseCacheAllocator("
            f"total={self.total_cache_ids}, "
            f"available={stats['available_cache_ids']}, "
            f"usage={stats['current_usage']}, "
            f"fragmentation={stats['fragmentation']:.2%})"
        )
