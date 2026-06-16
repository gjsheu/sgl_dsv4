from __future__ import annotations

from typing import Optional

import torch


def _verify_abs_positions(
    positions: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    n_draft: int,
) -> Optional[torch.Tensor]:
    request_num = positions.shape[0] // n_draft
    if request_num == 0:
        return None
    seq_lens_cpu = seq_lens_cpu[:request_num]
    if seq_lens_cpu.device.type != "cpu":
        seq_lens_cpu = seq_lens_cpu.cpu()
    start_positions = seq_lens_cpu - n_draft + 1
    return start_positions.view(-1, 1) + torch.arange(
        n_draft, dtype=start_positions.dtype
    ).view(1, -1)


def fill_verify_positions_gather(
    positions: torch.Tensor,
    dst: torch.Tensor,
    ratio: int,
    seq_lens_cpu: torch.Tensor,
    compress_ratios,
    n_draft: int,
) -> None:
    dst.zero_()
    if ratio not in compress_ratios or positions.numel() == 0:
        return

    abs_positions = _verify_abs_positions(positions, seq_lens_cpu, n_draft)
    if abs_positions is None:
        return

    boundary_mask = abs_positions % ratio == 0
    indices = torch.argsort((~boundary_mask).flatten(), dim=0, stable=True)
    indices = (
        indices[: min(positions.shape[0], dst.numel())]
        .pin_memory()
        .to(device=positions.device, non_blocking=True)
    )
    dst[: indices.numel()].copy_(torch.gather(positions, 0, indices))


def fill_verify_positions_boundary(
    positions: torch.Tensor,
    dst: torch.Tensor,
    ratio: int,
    seq_lens_cpu: torch.Tensor,
    compress_ratios,
    n_draft: int,
) -> None:
    dst.zero_()
    if ratio not in compress_ratios or positions.numel() == 0:
        return

    abs_positions = _verify_abs_positions(positions, seq_lens_cpu, n_draft)
    if abs_positions is None:
        return

    boundary_mask = abs_positions % ratio == 0
    indices = torch.nonzero(boundary_mask.flatten(), as_tuple=False).flatten()
    if indices.numel() == 0:
        return
    indices = indices[: dst.numel()].pin_memory().to(
        device=positions.device, non_blocking=True
    )
    dst[: indices.numel()].copy_(torch.gather(positions, 0, indices))


def build_compress_locs(
    *,
    page_size: int,
    compress_ratios,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    is_decode: bool,
    bs: int,
    device: torch.device,
    req_to_token_pool,
    out_cache_loc_dsv4,
    is_graph: bool = False,
) -> dict:
    result: dict = {}
    req_pool = req_pool_indices

    seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
    n_pages = max(1, (seq_lens_max + page_size - 1) // page_size)

    for ratio in compress_ratios:
        if ratio not in (4, 128):
            continue

        state_table = (
            req_to_token_pool.req_to_token_c4_state
            if ratio == 4
            else req_to_token_pool.req_to_token_c128_state
        )
        state_slots_2d = state_table[req_pool.to(torch.int64), : n_pages * page_size]
        state_page_2d = (state_slots_2d[:, ::page_size] // page_size).to(torch.int32)

        if is_decode:
            if out_cache_loc_dsv4 is None:
                raise RuntimeError(
                    "DSV4 decode metadata requires out_cache_loc_dsv4 from "
                    "DSV4NPUTokenToKVPoolAllocator.alloc_decode."
                )
            state_loc_decode = (
                out_cache_loc_dsv4.out_c4_state_loc
                if ratio == 4
                else out_cache_loc_dsv4.out_c128_state_loc
            ).to(torch.int32)

            compress_out_loc = torch.zeros(bs, dtype=torch.int32, device=device)
            bundle_loc = (
                out_cache_loc_dsv4.out_c4_loc
                if ratio == 4
                else out_cache_loc_dsv4.out_c128_loc
            )
            n_compress = bundle_loc.numel()
            if n_compress > 0:
                compress_out_loc[:n_compress] = bundle_loc.to(torch.int32)

        result[f"c{ratio}_state_page_table"] = state_page_2d
        if is_decode:
            result[f"c{ratio}_state_loc"] = state_loc_decode
            result[f"c{ratio}_loc"] = compress_out_loc

        c_table = (
            req_to_token_pool.req_to_token_c4
            if ratio == 4
            else req_to_token_pool.req_to_token_c128
        )
        n_c_tokens = seq_lens_max // ratio if is_graph else max(1, seq_lens_max // ratio)
        slots = c_table[req_pool.to(torch.int64), :n_c_tokens]
        result[f"c{ratio}_page_table"] = (slots[:, ::page_size] // page_size).to(
            torch.int32
        )

    if is_decode:
        valid = seq_lens > 0
        positions_last = torch.clamp(seq_lens - 1, min=0)
        for ratio in compress_ratios:
            if ratio not in (4, 128):
                continue
            padding_size = min(bs, bs // ratio + bs)
            padding = torch.zeros(padding_size, dtype=torch.int64, device=device)
            should_compress = ((seq_lens % ratio) == 0) & valid
            pos_cmp = positions_last[should_compress].to(torch.int64) + (1 - ratio)
            if pos_cmp.numel() > 0:
                padding[: pos_cmp.shape[0]].copy_(pos_cmp)
            result[f"positions_cmp_padding_c{ratio}"] = padding

        result["start_pos"] = positions_last.to(torch.int32)
        result["seqused"] = valid.to(torch.int32)

    return result
