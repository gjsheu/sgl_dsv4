from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class DeepseekV4AscendMultiStepDraftBackend:
    """Wrap DeepSeek-V4 Ascend attention backends for draft decode steps."""

    def __init__(
        self,
        model_runner: "ModelRunner",
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            DeepseekV4AscendAttnBackend(model_runner, speculative_step_id=step_id)
            for step_id in range(speculative_num_steps)
        ]

    def common_template(self, forward_batch: "ForwardBatch", call_fn):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def _step_out_cache_loc_dsv4(self, forward_batch: "ForwardBatch", step_id: int):
        bundle = forward_batch.out_cache_loc_dsv4
        if bundle is None or forward_batch.out_cache_loc is None:
            return None

        step_width = forward_batch.batch_size * self.topk
        total_width = step_width * self.speculative_num_steps
        raw_total_width = bundle.out_full_loc.numel()
        if (
            raw_total_width < total_width
            and raw_total_width % self.speculative_num_steps == 0
            and (raw_total_width // self.speculative_num_steps) % self.topk == 0
        ):
            # Graph replay pads forward_batch.batch_size up to the captured bs,
            # but the DSV4 loc bundle still contains only raw requests. Slice
            # with the raw width, then let metadata replay zero-fill graph tails.
            step_width = raw_total_width // self.speculative_num_steps
            total_width = raw_total_width
        if step_width == 0 or bundle.out_full_loc.numel() < total_width:
            return bundle

        full_steps = bundle.out_full_loc[:total_width].reshape(
            step_width // self.topk, self.topk, self.speculative_num_steps
        )
        full_steps = full_steps.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        swa_steps = bundle.out_swa_loc[:total_width].reshape(
            step_width // self.topk, self.topk, self.speculative_num_steps
        )
        swa_steps = swa_steps.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        def step_state(loc):
            if loc is None or loc.numel() < total_width:
                return loc
            steps = loc[:total_width].reshape(
                step_width // self.topk, self.topk, self.speculative_num_steps
            )
            return steps.permute((2, 0, 1)).reshape(
                self.speculative_num_steps, -1
            )[step_id]

        def step_compress(loc, ratio: int):
            if loc is None or loc.numel() == 0:
                return loc
            raw_bs = step_width // self.topk
            seq_lens = forward_batch.seq_lens[:raw_bs].to(torch.int64)
            positions = seq_lens[:, None, None] + torch.arange(
                self.speculative_num_steps,
                device=seq_lens.device,
                dtype=seq_lens.dtype,
            )
            positions = positions.expand(-1, self.topk, -1)
            should_compress = ((positions + 1) % ratio) == 0
            counts = should_compress.reshape(-1).to(torch.int64)
            offsets = torch.cumsum(counts, dim=0) - counts
            step_mask = should_compress[:, :, step_id].reshape(-1)
            step_offsets = offsets.reshape(
                raw_bs, self.topk, self.speculative_num_steps
            )[:, :, step_id].reshape(-1)
            return loc[step_offsets[step_mask].to(torch.int64)]

        return DSV4OutCacheLoc(
            out_full_loc=full_steps[step_id],
            out_swa_loc=swa_steps[step_id],
            out_c4_loc=step_compress(bundle.out_c4_loc, 4),
            out_c128_loc=step_compress(bundle.out_c128_loc, 128),
            out_c4_state_loc=step_state(bundle.out_c4_state_loc),
            out_c128_state_loc=step_state(bundle.out_c128_state_loc),
        )

    def _with_step_cache_locs(
        self, forward_batch: "ForwardBatch", step_id: int, call_fn
    ):
        old_out_cache_loc_dsv4 = forward_batch.out_cache_loc_dsv4
        forward_batch.out_cache_loc_dsv4 = self._step_out_cache_loc_dsv4(
            forward_batch, step_id
        )
        try:
            return call_fn()
        finally:
            forward_batch.out_cache_loc_dsv4 = old_out_cache_loc_dsv4

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        def call_fn(i, forward_batch):
            self._with_step_cache_locs(
                forward_batch,
                i,
                lambda: self.attn_backends[i].init_forward_metadata(forward_batch),
            )

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: "ForwardBatch"):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: "ForwardBatch", bs: int
    ):
        def call_fn(i, forward_batch):
            old_oc = forward_batch.out_cache_loc
            old_bundle = forward_batch.out_cache_loc_dsv4
            step_bundle = self._step_out_cache_loc_dsv4(forward_batch, i)
            forward_batch.out_cache_loc_dsv4 = step_bundle
            if (
                step_bundle is not None
                and step_bundle is not old_bundle
                and step_bundle.out_full_loc is not None
            ):
                forward_batch.out_cache_loc = step_bundle.out_full_loc
            self.attn_backends[i]._replay_forward_batch = forward_batch
            try:
                self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                    bs,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    seq_lens_sum=-1,
                    encoder_lens=None,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                )
            finally:
                self.attn_backends[i]._replay_forward_batch = None
                forward_batch.out_cache_loc = old_oc
                forward_batch.out_cache_loc_dsv4 = old_bundle

        self.common_template(forward_batch, call_fn)
