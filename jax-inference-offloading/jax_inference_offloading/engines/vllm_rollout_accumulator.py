#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thread-safe rollout result accumulator for async JAX-vLLM workflows.

This module provides a thread-safe accumulator for collecting streamed rollout
results from vLLM. It's designed for asynchronous architectures where:

- A consumer thread reads RolloutResult messages from a gRPC stream
- The main thread processes completed groups and may block on NCCL transfers
- Results must not be lost even when the main thread blocks

Usage:
    accumulator = RolloutAccumulator(num_rollouts=4)
    
    # Start consumer thread
    consumer = threading.Thread(
        target=result_consumer_thread,
        args=(results_stream, accumulator),
        daemon=True,
    )
    consumer.start()
    
    # Main loop
    while not done:
        groups = accumulator.get_completed_groups(timeout=0.1)
        for batch_id, prompt_idx, rollouts in groups:
            process_group(rollouts)
            if should_update_weights:
                transfer_engine.update_weights(params)  # May block
"""

import threading
import traceback
from collections import defaultdict
from queue import Queue, Empty

import jax_inference_offloading.api.controller_pb2 as ctrl


class RolloutAccumulator:
    """Thread-safe accumulator for rollout results.
    
    A consumer thread adds results via add_result(). The main thread
    retrieves completed groups via get_completed_groups(). This decoupling
    ensures that blocking operations (like NCCL weight transfers) in the
    main thread don't prevent result consumption.
    """

    def __init__(self, num_rollouts: int):
        """Initialize the accumulator.
        
        Args:
            num_rollouts: Number of rollouts per prompt (group is complete
                when this many results are accumulated for a (batch_id, prompt_idx) key).
        """
        self.num_rollouts = num_rollouts
        self._lock = threading.Lock()
        self._pending_groups = defaultdict(list)  # (batch_id, prompt_idx) -> [results]
        self._completed_queue = Queue()  # Thread-safe queue of completed groups
        self._stop_event = threading.Event()
        self._error = None  # Store any error from consumer thread

    def add_result(self, result):
        """Add a rollout result (called from consumer thread).
        
        When a group reaches num_rollouts results, it's moved to the
        completed queue for the main thread to retrieve.
        
        Args:
            result: A RolloutResult protobuf message.
        """
        key = (result.batch_id, result.prompt_index)
        with self._lock:
            self._pending_groups[key].append(result)
            if len(self._pending_groups[key]) == self.num_rollouts:
                group = self._pending_groups.pop(key)
                # Put completed group in queue: (batch_id, prompt_idx, [results])
                self._completed_queue.put((result.batch_id, result.prompt_index, group))

    def get_completed_groups(self, max_groups=None, timeout=0.1):
        """Get completed groups (called from main thread).
        
        This is non-blocking (uses timeout). Returns whatever groups are
        available, up to max_groups.
        
        Args:
            max_groups: Maximum number of groups to return. None means all available.
            timeout: Timeout in seconds for waiting on first group.
        
        Returns:
            List of (batch_id, prompt_idx, [results]) tuples.
        """
        groups = []
        try:
            # Wait for at least one group (with timeout)
            first = self._completed_queue.get(timeout=timeout)
            groups.append(first)
            
            # Get any additional groups that are ready (non-blocking)
            while max_groups is None or len(groups) < max_groups:
                try:
                    group = self._completed_queue.get_nowait()
                    groups.append(group)
                except Empty:
                    break
        except Empty:
            pass  # No groups available within timeout
        
        return groups

    def stop(self):
        """Signal the consumer thread to stop."""
        self._stop_event.set()

    @property
    def should_stop(self):
        """Check if stop was requested."""
        return self._stop_event.is_set()

    def set_error(self, error):
        """Store an error from the consumer thread."""
        self._error = error

    @property
    def error(self):
        """Get any error from the consumer thread."""
        return self._error

    @property
    def pending_count(self):
        """Get number of pending (incomplete) groups."""
        with self._lock:
            return len(self._pending_groups)

    @property
    def completed_queue_size(self):
        """Get number of completed groups waiting to be processed."""
        return self._completed_queue.qsize()


def result_consumer_thread(results_stream, accumulator, verbose=False):
    """Consumer thread that reads from gRPC stream and accumulates results.
    
    This thread runs independently of the main thread, continuously reading
    from the gRPC stream. Even when the main thread blocks on NCCL transfers,
    this thread keeps consuming results, preventing message loss.
    
    Args:
        results_stream: gRPC subscription stream for RolloutResult messages.
        accumulator: RolloutAccumulator instance for storing results.
        verbose: If True, print each received result.
    """
    try:
        for delivery in results_stream:
            if accumulator.should_stop:
                break
            
            result = ctrl.RolloutResult()
            delivery.message.payload.Unpack(result)
            
            if verbose:
                print(
                    f"[Consumer] Received: batch={result.batch_id[:8]}..., "
                    f"prompt={result.prompt_index}, rollout={result.rollout_index}"
                )
            
            accumulator.add_result(result)
            
    except Exception as e:
        import grpc
        if isinstance(e, grpc.RpcError) and e.code() in (
            grpc.StatusCode.CANCELLED, grpc.StatusCode.UNAVAILABLE
        ):
            # Expected when shutting down
            pass
        else:
            print(f"[Consumer] Error: {e}")
            traceback.print_exc()
            accumulator.set_error(e)
