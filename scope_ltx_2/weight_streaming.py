"""Weight streaming for memory-efficient transformer inference.

Streams transformer blocks between CPU (pinned memory) and GPU using
double-buffered CUDA streams to overlap data transfer with computation.

Architecture:
    - All blocks start on CPU in pinned memory (allocated once at setup)
    - Pre-forward hook loads block N to GPU on the transfer stream
    - While block N computes on the default stream, block N+1 is prefetched
    - Post-forward hook schedules async offload of block N-1
    - Two CUDA streams alternate: one transfers while the other computes
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BlockStreamingConfig:
    blocks_to_stream: int = 0
    prefetch_blocks: int = 1
    use_pinned_memory: bool = True
    offload_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    compute_device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.compute_device is None:
            self.compute_device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )


class BlockStreamingState:
    """Manages double-buffered block streaming during inference."""

    def __init__(self, blocks: nn.ModuleList, config: BlockStreamingConfig) -> None:
        self.blocks = blocks
        self.config = config
        self.num_blocks = len(blocks)
        self.num_resident = max(0, self.num_blocks - config.blocks_to_stream)
        self.streaming_start_idx = self.num_resident
        self.block_on_gpu: list[bool] = [False] * self.num_blocks

        self._transfer_stream: torch.cuda.Stream | None = None
        self._offload_stream: torch.cuda.Stream | None = None

        self._pinned_params: list[dict[str, torch.Tensor]] = [{} for _ in range(self.num_blocks)]

    def _ensure_streams(self) -> None:
        if self._transfer_stream is None and torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream(device=self.config.compute_device)
            self._offload_stream = torch.cuda.Stream(device=self.config.compute_device)

    def pin_block(self, block_idx: int) -> None:
        """Pin a block's parameters to page-locked memory (done once at setup)."""
        block = self.blocks[block_idx]
        pinned = {}
        for name, param in block.named_parameters():
            if param.device.type != "cpu":
                param.data = param.data.cpu()
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
            if not param.data.is_pinned():
                try:
                    p = torch.empty_like(param.data, pin_memory=True)
                    p.copy_(param.data)
                    param.data = p
                except RuntimeError:
                    pass
            pinned[name] = param.data
        self._pinned_params[block_idx] = pinned

    def load_block_to_gpu(self, block_idx: int, stream: torch.cuda.Stream | None = None) -> None:
        """Transfer block parameters from pinned CPU memory to GPU."""
        if self.block_on_gpu[block_idx]:
            return
        block = self.blocks[block_idx]
        device = self.config.compute_device
        ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            for param in block.parameters():
                if param.device != device:
                    param.data = param.data.to(device, non_blocking=True)
            for buf in block.buffers():
                if buf.device != device:
                    buf.data = buf.data.to(device, non_blocking=True)
        self.block_on_gpu[block_idx] = True

    def offload_block_to_cpu(self, block_idx: int, stream: torch.cuda.Stream | None = None) -> None:
        """Move block back to its pinned CPU copy."""
        if not self.block_on_gpu[block_idx]:
            return
        if block_idx < self.streaming_start_idx:
            return
        block = self.blocks[block_idx]
        pinned = self._pinned_params[block_idx]
        ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            for name, param in block.named_parameters():
                if name in pinned:
                    param.data = pinned[name]
                else:
                    param.data = param.data.to(self.config.offload_device, non_blocking=True)
            for buf in block.buffers():
                if buf.device.type != "cpu":
                    buf.data = buf.data.to(self.config.offload_device, non_blocking=True)
        self.block_on_gpu[block_idx] = False

    def cleanup(self) -> None:
        if self._transfer_stream is not None:
            self._transfer_stream.synchronize()
        if self._offload_stream is not None:
            self._offload_stream.synchronize()
        self._transfer_stream = None
        self._offload_stream = None
        self._pinned_params = [{} for _ in range(self.num_blocks)]
        self.blocks = None


class _nullcontext:
    def __enter__(self) -> None:
        return None
    def __exit__(self, *args: Any) -> None:
        pass


def _create_pre_hook(state: BlockStreamingState, block_idx: int):
    def hook(module: nn.Module, inputs: tuple) -> None:
        state._ensure_streams()

        if not state.block_on_gpu[block_idx]:
            state._transfer_stream.synchronize()
            state.load_block_to_gpu(block_idx, stream=state._transfer_stream)
            state._transfer_stream.synchronize()

        torch.cuda.current_stream().wait_stream(state._transfer_stream)

        for i in range(1, state.config.prefetch_blocks + 1):
            nxt = block_idx + i
            if nxt < state.num_blocks and not state.block_on_gpu[nxt]:
                state._transfer_stream.wait_stream(torch.cuda.current_stream())
                state.load_block_to_gpu(nxt, stream=state._transfer_stream)

    return hook


def _create_post_hook(state: BlockStreamingState, block_idx: int):
    def hook(module: nn.Module, inputs: tuple, outputs: Any) -> None:
        prev = block_idx - 1
        if prev >= state.streaming_start_idx and state.block_on_gpu[prev]:
            state._offload_stream.wait_stream(torch.cuda.current_stream())
            state.offload_block_to_cpu(prev, stream=state._offload_stream)

        if block_idx == state.num_blocks - 1 and state.block_on_gpu[block_idx]:
            if block_idx >= state.streaming_start_idx:
                state._offload_stream.wait_stream(torch.cuda.current_stream())
                state.offload_block_to_cpu(block_idx, stream=state._offload_stream)

    return hook


def setup_block_streaming(
    blocks: nn.ModuleList,
    config: BlockStreamingConfig,
) -> BlockStreamingState:
    num_blocks = len(blocks)
    config.blocks_to_stream = max(0, min(config.blocks_to_stream, num_blocks - 1))
    state = BlockStreamingState(blocks, config)
    device = config.compute_device

    if config.blocks_to_stream <= 0:
        logger.info(f"All {num_blocks} blocks fit on GPU — no streaming needed")
        t0 = time.time()
        for i in range(num_blocks):
            block = blocks[i]
            for param in block.parameters():
                if param.device != device:
                    param.data = param.data.to(device, non_blocking=True)
            for buf in block.buffers():
                if buf.device != device:
                    buf.data = buf.data.to(device, non_blocking=True)
            state.block_on_gpu[i] = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"All blocks on GPU: {allocated:.2f} GB allocated ({time.time()-t0:.1f}s)")
        return state

    logger.info(
        f"Setting up block streaming: {state.num_resident} resident, "
        f"{config.blocks_to_stream} streaming, prefetch={config.prefetch_blocks}"
    )

    t0 = time.time()
    if config.use_pinned_memory:
        for i in range(state.streaming_start_idx, num_blocks):
            state.pin_block(i)
        logger.info(f"Pinned {config.blocks_to_stream} blocks in {time.time()-t0:.1f}s")
    else:
        for i in range(state.streaming_start_idx, num_blocks):
            block = blocks[i]
            for param in block.parameters():
                if param.device.type != "cpu":
                    param.data = param.data.cpu()
            for buf in block.buffers():
                if buf.device.type != "cpu":
                    buf.data = buf.data.cpu()
            state.block_on_gpu[i] = False

    for i in range(state.streaming_start_idx):
        block = blocks[i]
        for param in block.parameters():
            if param.device != device:
                param.data = param.data.to(device, non_blocking=True)
        for buf in block.buffers():
            if buf.device != device:
                buf.data = buf.data.to(device, non_blocking=True)
        state.block_on_gpu[i] = True

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for i in range(num_blocks):
        blocks[i].register_forward_pre_hook(_create_pre_hook(state, i))
        blocks[i].register_forward_hook(_create_post_hook(state, i))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"Block streaming ready: {allocated:.2f} GB allocated")

    return state


def cleanup_block_streaming(
    blocks: nn.ModuleList,
    state: BlockStreamingState | None = None,
    move_to: str | torch.device | None = None,
) -> None:
    for block in blocks:
        block._forward_pre_hooks.clear()
        block._forward_hooks.clear()

    if state is not None:
        state.cleanup()

    if move_to is not None:
        target = torch.device(move_to)
        for block in blocks:
            for param in block.parameters():
                if param.device != target:
                    param.data = param.data.to(target, non_blocking=True)
                elif target.type == "cpu" and param.data.is_pinned():
                    param.data = param.data.clone()
            for buf in block.buffers():
                if buf.device != target:
                    buf.data = buf.data.to(target, non_blocking=True)
                elif target.type == "cpu" and buf.data.is_pinned():
                    buf.data = buf.data.clone()
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_block_memory(block: nn.Module) -> int:
    total = 0
    for param in block.parameters():
        total += param.numel() * param.element_size()
    for buf in block.buffers():
        total += buf.numel() * buf.element_size()
    return total


def calculate_optimal_streaming_config(
    blocks: nn.ModuleList,
    available_vram_gb: float,
    safety_margin_gb: float = 1.5,
    min_resident_blocks: int = 4,
) -> BlockStreamingConfig:
    num_blocks = len(blocks)
    if num_blocks == 0:
        return BlockStreamingConfig(blocks_to_stream=0)

    prefetch_blocks = 2

    block_sizes = [estimate_block_memory(b) for b in blocks]
    budget_bytes = (available_vram_gb - safety_margin_gb) * 1024**3

    if sum(block_sizes) <= budget_bytes:
        logger.info(f"All {num_blocks} blocks fit on GPU ({sum(block_sizes)/1024**3:.1f} GB), no streaming")
        return BlockStreamingConfig(blocks_to_stream=0, prefetch_blocks=prefetch_blocks)

    # During the forward pass, up to (prefetch + 1) streaming blocks may be on
    # GPU concurrently (current block + prefetch slots), plus 1 extra to account
    # for async offload latency before the previous block's memory is reclaimed.
    peak_concurrent_streaming = prefetch_blocks + 2

    # Find the maximum number of resident blocks such that
    # resident_mem + peak_streaming_mem fits within the budget.
    # Streaming blocks are the tail of the list; peak streaming memory is
    # the sum of the largest `peak_concurrent_streaming` blocks in that tail.
    max_resident = min_resident_blocks
    for n in range(num_blocks, min_resident_blocks - 1, -1):
        resident_mem = sum(block_sizes[:n])
        streaming_tail = block_sizes[n:]
        if not streaming_tail:
            peak_streaming = 0
        else:
            concurrent = min(peak_concurrent_streaming, len(streaming_tail))
            peak_streaming = sum(sorted(streaming_tail, reverse=True)[:concurrent])
        if resident_mem + peak_streaming <= budget_bytes:
            max_resident = n
            break

    blocks_to_stream = num_blocks - max_resident

    resident_gb = sum(block_sizes[:max_resident]) / 1024**3
    stream_gb = sum(block_sizes[max_resident:]) / 1024**3
    logger.info(
        f"Streaming config: {num_blocks} blocks, "
        f"{max_resident} resident ({resident_gb:.1f} GB), "
        f"{blocks_to_stream} streaming ({stream_gb:.1f} GB), "
        f"budget={budget_bytes/1024**3:.1f} GB (safety={safety_margin_gb} GB)"
    )

    return BlockStreamingConfig(
        blocks_to_stream=blocks_to_stream,
        prefetch_blocks=prefetch_blocks,
        use_pinned_memory=True,
    )
