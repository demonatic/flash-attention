# mask mod test script
# REFACTORED to use _flash_attn_fwd as the kernel entrypoint
#
# Test Organization:
# - test_static_masks: Fast tests for masks that don't need per-seqlen compilation
#   (identity, document, block_diagonal, etc.) with comprehensive seqlen coverage
# - test_parameterized_masks: Slower tests for masks that require recompilation per
#   seqlen pair (causal, block_causal, sliding_window) with reduced seqlen coverage
#
# Usage:
#   pytest test_mask_mod.py::test_static_masks         # Run only fast tests
#   pytest test_mask_mod.py::test_parameterized_masks  # Run only slow tests
#   pytest test_mask_mod.py                            # Run all tests

import math
from typing import Optional
from einops import rearrange, repeat

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.interface import flash_attn_func, _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch, bhqk_to_linear_sparse_tensors, LinearBlockSparseTensorsTorch

# Import CUDA kernel for create_block_mask
try:
    import create_block_mask_cuda
    HAS_CREATE_BLOCK_MASK_CUDA = True
except ImportError:
    # Try to build and install create_block_mask_cuda
    import subprocess
    import os
    print("create_block_mask_cuda not found. Attempting to build and install...")
    utils_dir = os.path.join(os.path.dirname(__file__), "../../csrc/utils")
    utils_dir = os.path.abspath(utils_dir)
    try:
        # Run make to build the module
        result = subprocess.run(
            ["make", "create_block_mask"],
            cwd=utils_dir,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Build successful. Importing create_block_mask_cuda...")
            import create_block_mask_cuda
            HAS_CREATE_BLOCK_MASK_CUDA = True
        else:
            print(f"Build failed: {result.stderr}")
            create_block_mask_cuda = None
            HAS_CREATE_BLOCK_MASK_CUDA = False
    except Exception as e:
        print(f"Failed to build create_block_mask_cuda: {e}")
        create_block_mask_cuda = None
        HAS_CREATE_BLOCK_MASK_CUDA = False
        print("Warning: create_block_mask_cuda not found. Using PyTorch reference implementation.")
from flash_attn.cute.mask_definitions import (
    get_mask_pair,
    STATIC_MASKS,
    random_arbitrary_func_tensor,
)
COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


# def get_mask(arbitrary_func, seqlen_q, seqlen_k):
#     func_num = arbitrary_func.shape[2]
#     mask = torch.ones(1, 1, seqlen_q, seqlen_k).cuda()
#     for i in range(seqlen_q):
#         for j in range(seqlen_k):
#             value_valid = j < arbitrary_func[0, 0, 0, i]
#             for k in range(func_num // 2):
#                 if j >= arbitrary_func[0, 0, 2 * k + 1, i] and j < arbitrary_func[0, 0, 2 * k + 2, i]:
#                     value_valid = True
#             mask[:, :, i, j] = False if not value_valid else True
#     # print("mask is {}".format(mask))
#     return mask

import torch

import torch

def apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k):
    """
    qk_attn: [B, H, seqlen_q, seqlen_k]
    arbitrary_func: [1, 1, func_num, seqlen_q + 256] (int-like indices)
    Only the first `seqlen_q` positions of the last dim are used.
    """
    device = qk_attn.device
    B, H, sq, sk = qk_attn.shape
    assert sq == seqlen_q and sk == seqlen_k

    # Use only the first seqlen_q entries along last dim
    af = arbitrary_func[..., :seqlen_q]   # [1,1,func_num,seqlen_q]
    func_num = af.shape[2]

    # j grid: [1,1,1,seqlen_k]
    j = torch.arange(seqlen_k, device=device, dtype=af.dtype).view(1, 1, 1, seqlen_k)

    # base cutoff: af[0,0,0,i] -> [1,1,seqlen_q,1]
    base_cut = af[0, 0, 0, :].view(1, 1, seqlen_q, 1)
    base_valid = j < base_cut   # [1,1,seqlen_q,seqlen_k]

    # intervals: starts = af[0,0,1::2,:], ends = af[0,0,2::2,:]
    num_intervals = func_num // 2
    if num_intervals > 0:
        starts = af[0, 0, 1:func_num:2, :]  # [num_intervals, seqlen_q]
        ends   = af[0, 0, 2:func_num:2, :]  # [num_intervals, seqlen_q]

        # reshape for broadcasting: [1, num_intervals, seqlen_q, 1]
        starts = starts.view(1, num_intervals, seqlen_q, 1)
        ends   = ends.view(1, num_intervals, seqlen_q, 1)

        # j expanded: [1,1,seqlen_q,seqlen_k] -> then [1, num_intervals, seqlen_q, seqlen_k]
        j_exp = j.expand(1, num_intervals, seqlen_q, seqlen_k)

        in_interval = (j_exp >= starts) & (j_exp < ends)  # [1,num_intervals,seqlen_q,seqlen_k]
        interval_valid = in_interval.any(dim=1, keepdim=True)  # [1,1,seqlen_q,seqlen_k]
    else:
        interval_valid = torch.zeros(1, 1, seqlen_q, seqlen_k, dtype=torch.bool, device=device)

    # combined mask [1,1,seqlen_q,seqlen_k]
    valid = base_valid | interval_valid

    # expand to [B,H,seqlen_q,seqlen_k]
    valid = valid.expand(B, H, seqlen_q, seqlen_k)

    # set invalid positions to -inf (works in-place style by assigning back)
    qk_attn = torch.where(valid, qk_attn, torch.full_like(qk_attn, -float("inf")))

    return qk_attn





def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    # lengths = torch.randint(1, seqlen_q + 1, (batch_size,))
    # cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    # cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    # total_q = cu_seqlens_q[-1]
    # total_k = total_q
    q = torch.empty(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
    k = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    ).uniform_(-1, 1).requires_grad_(True)
    v = torch.empty(
        batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    ).uniform_(-1, 1).requires_grad_(True)

    # q = torch.ones(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype).requires_grad_(True)
    # k = torch.ones(
    #     batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    # ).requires_grad_(True)
    # v = torch.ones(
    #     batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    # ).requires_grad_(True)

    out = torch.empty(
        batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, nheads, seqlen_q, device=device, dtype=torch.float32)

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "out": out.contiguous(),
        "lse": lse.contiguous(),
        "cu_seqlens_q": None,
        "cu_seqlens_k": None,
    }

def compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False):
    """Compute reference of arbitrary mask"""
    q = tensors["q"] if not up_cast else tensors["q"].float()
    k = tensors["k"] if not up_cast else tensors["k"].float()
    v = tensors["v"] if not up_cast else tensors["v"].float()
    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    nheads = q.shape[2]
    nheads_kv = k.shape[2]
    headdim = q.shape[3]
    scale = 1.0 / math.sqrt(headdim)

    if nheads_kv == nheads:
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q * scale,
            k,
        )
    else:
        k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum(
            "bnhd,bmhd->bhnm",
            q * scale,
            k,
        )

    # func_num = arbitrary_func.shape[2]
    # for i in range(seqlen_q):
    #     for j in range(seqlen_k):
    #         value_valid = j < arbitrary_func[0, 0, 0, i]
    #         for k in range(func_num // 2):
    #             if j >= arbitrary_func[0, 0, 2 * k + 1, i] and j < arbitrary_func[0, 0, 2 * k + 2, i]:
    #                 value_valid = True
    #         qk_attn[:, :, i, j] = -float("inf") if not value_valid else qk_attn[:, :, i, j]
    qk_attn = apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k)

    all_inf_mask = torch.all(qk_attn == float('-inf'), dim=-1, keepdim=True)  # [B, H, seqlen_q, 1]
    softmax_attn = F.softmax(qk_attn, dim=-1)
    softmax_attn = torch.where(all_inf_mask, torch.zeros_like(softmax_attn), softmax_attn)
    out = torch.einsum(
        "bhnm,bmhd->bnhd",
        softmax_attn,
        v,
    )

    is_all_zero = torch.count_nonzero(arbitrary_func) == 0
    if is_all_zero:
        out[:] = 0.0

    return out


def compute_reference_block_max_scores(tensors, arbitrary_func=None):
    """Per (batch, head, q_pos), max of QK (optionally arbitrary-masked) per K-tile of 128.

    Matches SM100 ``max_score_out``: raw QK^T **before** ``1/sqrt(d)`` softmax scale, BF16
    GEMM accumulated in FP32; optional ``apply_arbitrary_mask_to_qk`` when ``arbitrary_func``
    is set (note: may diverge from kernel for dense arbitrary; use ``None`` for strict checks).
    """
    q = tensors["q"]
    k = tensors["k"]
    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    nheads = q.shape[2]
    nheads_kv = k.shape[2]

    if nheads_kv == nheads:
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k).to(torch.float32)
    else:
        k_exp = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k_exp).to(torch.float32)

    if arbitrary_func is not None:
        qk_attn = apply_arbitrary_mask_to_qk(
            qk_attn, arbitrary_func, seqlen_q, seqlen_k
        )

    pad = (128 - (seqlen_k % 128)) % 128
    if pad:
        qk_attn = F.pad(qk_attn, (0, pad), value=float("-inf"))
    num_chunks = qk_attn.shape[-1] // 128
    qk_chunks = qk_attn.view(batch_size, nheads, seqlen_q, num_chunks, 128)
    return qk_chunks.amax(dim=-1).contiguous()


def compute_reference_block_lse(tensors, softmax_scale, arbitrary_func=None):
    """Per (batch, head, q_pos), scaled LSE per K-tile of 128.

    Computes ``log(sum(exp(QK * softmax_scale)))`` per K block, matching the kernel's
    ``block_lse_out``.  Uses BF16 QK GEMM accumulated in FP32.
    """
    q = tensors["q"]
    k = tensors["k"]
    batch_size = q.shape[0]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    nheads = q.shape[2]
    nheads_kv = k.shape[2]

    if nheads_kv == nheads:
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k).to(torch.float32)
    else:
        k_exp = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k_exp).to(torch.float32)

    if arbitrary_func is not None:
        qk_attn = apply_arbitrary_mask_to_qk(
            qk_attn, arbitrary_func, seqlen_q, seqlen_k
        )

    qk_attn = qk_attn * softmax_scale

    pad = (128 - (seqlen_k % 128)) % 128
    if pad:
        qk_attn = F.pad(qk_attn, (0, pad), value=float("-inf"))
    num_chunks = qk_attn.shape[-1] // 128
    qk_chunks = qk_attn.view(batch_size, nheads, seqlen_q, num_chunks, 128)
    return torch.logsumexp(qk_chunks, dim=-1).contiguous()


def _get_per_doc_info(arbitrary_func, seqlen_q, seqlen_k, block_size_k=128):
    """Extract per-doc K-block info from arbitrary_func.

    Returns (doc_k_starts, n_per_doc_chunks) where:
      doc_k_starts: int tensor (seqlen_q,) — K start per Q row
      n_per_doc_chunks: int — buffer size matching kernel output (= global K-blocks)
    """
    af = arbitrary_func[0, 0, 1, :seqlen_q].long()
    n_chunks = (seqlen_k + block_size_k - 1) // block_size_k
    return af, n_chunks


def _apply_block_sparse_to_qk_tokens(
    qk_attn, linear_k, seqlen_q, seqlen_k, q_super_block, block_size_k=128,
):
    """Set QK positions to -inf for K tokens in unvisited global K-blocks.

    Unlike ``apply_block_sparse_unvisited_to_max_score`` (which works on block-level
    scores with global indexing), this operates on the token-level QK matrix so that
    per-doc block references can be computed on the doubly-masked result.
    """
    out = qk_attn.clone()
    num_qb = (seqlen_q + q_super_block - 1) // q_super_block
    num_kb = (seqlen_k + block_size_k - 1) // block_size_k
    for m in range(num_qb):
        q_lo = m * q_super_block
        q_hi = min((m + 1) * q_super_block, seqlen_q)
        s = int(linear_k.mask_block_offset[m].item())
        e = int(linear_k.mask_block_offset[m + 1].item())
        parts = [linear_k.mask_block_idx[s:e].long()]
        if linear_k.full_block_idx is not None and linear_k.full_block_offset is not None:
            fs = int(linear_k.full_block_offset[m].item())
            fe = int(linear_k.full_block_offset[m + 1].item())
            parts.append(linear_k.full_block_idx[fs:fe].long())
        visited = torch.cat(parts)
        visited = visited[(visited >= 0) & (visited < num_kb)]
        k_mask = torch.zeros(seqlen_k, dtype=torch.bool, device=out.device)
        for kb in visited:
            k_lo_pos = kb.item() * block_size_k
            k_hi_pos = min(k_lo_pos + block_size_k, seqlen_k)
            k_mask[k_lo_pos:k_hi_pos] = True
        out[:, :, q_lo:q_hi, ~k_mask] = float("-inf")
    return out


def compute_reference_per_doc_block_max_scores(
    tensors, arbitrary_func, block_size_k=128, linear_k=None, q_super_block=None,
):
    """Per-doc block max scores: K-blocks aligned to each Q row's doc_k_start.

    For Q row q with doc_k_start = arbitrary_func[0,0,1,q]:
      per-doc K-block b covers global K range [doc_k_start + b*128, doc_k_start + (b+1)*128).

    Returns: (batch, nheads, seqlen_q, n_per_doc_chunks), float32
    """
    q, k = tensors["q"], tensors["k"]
    batch_size, seqlen_q, nheads, _ = q.shape
    seqlen_k = k.shape[1]
    nheads_kv = k.shape[2]

    if nheads_kv == nheads:
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k).to(torch.float32)
    else:
        k_exp = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k_exp).to(torch.float32)

    qk_attn = apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k)
    if linear_k is not None:
        qk_attn = _apply_block_sparse_to_qk_tokens(
            qk_attn, linear_k, seqlen_q, seqlen_k, q_super_block, block_size_k,
        )

    doc_k_starts, n_chunks = _get_per_doc_info(
        arbitrary_func, seqlen_q, seqlen_k, block_size_k
    )

    result = torch.full(
        (batch_size, nheads, seqlen_q, n_chunks),
        float("-inf"), dtype=torch.float32, device=q.device,
    )
    for qi in range(seqlen_q):
        dk_start = doc_k_starts[qi].item()
        for b in range(n_chunks):
            k_lo = dk_start + b * block_size_k
            k_hi = min(k_lo + block_size_k, seqlen_k)
            if k_lo >= seqlen_k:
                break
            result[:, :, qi, b] = qk_attn[:, :, qi, k_lo:k_hi].amax(dim=-1)
    return result


def compute_reference_per_doc_block_lse(
    tensors, softmax_scale, arbitrary_func, block_size_k=128,
    linear_k=None, q_super_block=None,
):
    """Per-doc block LSE: K-blocks aligned to each Q row's doc_k_start.

    Same layout as ``compute_reference_per_doc_block_max_scores`` but computes
    ``log(sum(exp(QK * softmax_scale)))`` per per-doc K-block.

    Returns: (batch, nheads, seqlen_q, n_per_doc_chunks), float32
    """
    q, k = tensors["q"], tensors["k"]
    batch_size, seqlen_q, nheads, _ = q.shape
    seqlen_k = k.shape[1]
    nheads_kv = k.shape[2]

    if nheads_kv == nheads:
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k).to(torch.float32)
    else:
        k_exp = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_kv)
        qk_attn = torch.einsum("bnhd,bmhd->bhnm", q, k_exp).to(torch.float32)

    qk_attn = apply_arbitrary_mask_to_qk(qk_attn, arbitrary_func, seqlen_q, seqlen_k)
    if linear_k is not None:
        qk_attn = _apply_block_sparse_to_qk_tokens(
            qk_attn, linear_k, seqlen_q, seqlen_k, q_super_block, block_size_k,
        )
    qk_attn = qk_attn * softmax_scale

    doc_k_starts, n_chunks = _get_per_doc_info(
        arbitrary_func, seqlen_q, seqlen_k, block_size_k
    )

    result = torch.full(
        (batch_size, nheads, seqlen_q, n_chunks),
        float("-inf"), dtype=torch.float32, device=q.device,
    )
    for qi in range(seqlen_q):
        dk_start = doc_k_starts[qi].item()
        for b in range(n_chunks):
            k_lo = dk_start + b * block_size_k
            k_hi = min(k_lo + block_size_k, seqlen_k)
            if k_lo >= seqlen_k:
                break
            result[:, :, qi, b] = torch.logsumexp(
                qk_attn[:, :, qi, k_lo:k_hi], dim=-1,
            )
    return result


def apply_block_sparse_unvisited_to_max_score(
    ref_dense: torch.Tensor,
    linear_k: LinearBlockSparseTensorsTorch,
    seqlen_q: int,
    q_super_block: int,
) -> torch.Tensor:
    """Set ``max_score`` slots to ``-inf`` for K-blocks not in the forward sparse list per Q super-block.

    ``linear_k`` must match ``create_block_mask(..., BLOCK_SIZE=(q_super_block, 128))`` used for Q2K.
    """
    out = ref_dense.clone()
    _, _, _, n_chunks = out.shape
    num_qb = (seqlen_q + q_super_block - 1) // q_super_block
    assert int(linear_k.mask_block_offset.shape[0]) == num_qb + 1, (
        f"mask_block_offset len {linear_k.mask_block_offset.shape[0]} != num_qb+1={num_qb + 1}"
    )
    for m in range(num_qb):
        q_lo = m * q_super_block
        q_hi = min((m + 1) * q_super_block, seqlen_q)
        s = int(linear_k.mask_block_offset[m].item())
        e = int(linear_k.mask_block_offset[m + 1].item())
        mask_nb = linear_k.mask_block_idx[s:e].long()
        parts = [mask_nb]
        if linear_k.full_block_idx is not None and linear_k.full_block_offset is not None:
            fs = int(linear_k.full_block_offset[m].item())
            fe = int(linear_k.full_block_offset[m + 1].item())
            parts.append(linear_k.full_block_idx[fs:fe].long())
        visited = torch.cat(parts)
        visited = visited[(visited >= 0) & (visited < n_chunks)]
        vis = torch.zeros(n_chunks, dtype=torch.bool, device=out.device)
        vis[visited] = True
        out[:, :, q_lo:q_hi, ~vis] = float("-inf")
    return out


def _run_mask_test(
    seqlen_q,
    seqlen_k,
    nheads,
    kv_mode,
    headdim,
    dtype,
    tile_m,
    tile_n,
    use_block_sparsity,
    load_tensor=False,
    deterministic=False,
):
    # torch.manual_seed(42)

    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 1
    headdim_v = headdim

    # aux_tensors_arg = None
    # mask_mod_cute, mask_mod_flex = get_mask_pair("causal", seqlen_q, seqlen_k)
    mask_mod_cute, mask_mod_flex = get_mask_pair("arbitrary")
    arbitrary_func = random_arbitrary_func_tensor(1, 1, 3, seqlen_q, seqlen_k, device="cuda")


    if load_tensor:
        func_path = "jerry_func_2.pt"
        arbitrary_func = torch.load(func_path, map_location="cpu").cuda()


    original_flex_mask = mask_mod_flex

    causal = False

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    if load_tensor:
        q_path = "jerry_q_2.pt"
        k_path = "jerry_k_2.pt"
        v_path = "jerry_v_2.pt"

        tensors["q"] = torch.load(q_path, map_location="cpu").cuda().requires_grad_()  
        tensors["k"] = torch.load(k_path, map_location="cpu").cuda().requires_grad_() 
        tensors["v"] = torch.load(v_path, map_location="cpu").cuda().requires_grad_()   
    
    
    aux_tensors_arg = [arbitrary_func]

    frozen_af = arbitrary_func  # freeze
    def mask_mod_flex_new(b, h, q_idx, kv_idx, arbitrary_func_func=frozen_af):
        return original_flex_mask(b, h, q_idx, kv_idx, arbitrary_func_func)

    headdim = tensors["q"].shape[3]
    softmax_scale = 1.0 / math.sqrt(headdim)

    # Compute block sparsity for mask_mod
    if COMPUTE_CAPABILITY == 10:
        sparse_tile_m = 2 * tile_m
    else:
        sparse_tile_m = tile_m

    # =========================================================================
    # Method 1: Use PyTorch's create_block_mask (reference)
    # =========================================================================
    bm = create_block_mask(
        mask_mod_flex_new,
        1,
        1,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    # bm.as_tuple() format: (Q_LEN, KV_LEN, kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices,
    #                        q_num_blocks, q_indices, full_q_num_blocks, full_q_indices, Q_BLOCK_SIZE, KV_BLOCK_SIZE, mask_mod)
    _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm.as_tuple()
    
    k_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=k_mask_cnt,
        mask_block_idx=k_mask_idx,
        full_block_cnt=k_full_cnt,
        full_block_idx=k_full_idx,
    )
    # convert to linear sparse tensors kernel
    ref_linear_k_block_sparse_mask = bhqk_to_linear_sparse_tensors(k_block_sparse_mask)

    bm_bwd = create_block_mask(
        mask_mod_flex_new,
        1,
        1,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(tile_m, tile_n),
    )
    # bm_bwd.as_tuple() format: same as above, but we need q_* (indices 6-9)
    _, _, _, _, _, _, q_mask_cnt, q_mask_idx, q_full_cnt, q_full_idx, *_ = bm_bwd.as_tuple()
    q_block_sparse_mask = BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=q_full_cnt,
        full_block_idx=q_full_idx,
    )
    ref_linear_q_block_sparse_mask = bhqk_to_linear_sparse_tensors(q_block_sparse_mask)

    # =========================================================================
    # Method 2: Use CUDA kernel (create_block_mask_cuda)
    # =========================================================================
    if HAS_CREATE_BLOCK_MASK_CUDA:
        # nvtx tag
        with torch.cuda.nvtx.range("create_q2k_csr_sparse_from_func"):
            # Q2K (Forward): fix q_block, loop kv_blocks
            (cuda_k_mask_cnt, cuda_k_mask_offset, cuda_k_mask_idx,
            cuda_k_full_cnt, cuda_k_full_offset, cuda_k_full_idx) = \
                create_block_mask_cuda.create_q2k_csr_sparse_from_func(
                    arbitrary_func,
                    seqlen_q,
                    seqlen_k,
                    Q_BLOCK_SIZE=sparse_tile_m,
                    KV_BLOCK_SIZE=tile_n,
                    check_q_boundary=True  # True just to align to flex attention, change to False for performance
                )
            
            # Convert to LinearBlockSparseTensorsTorch format
            # CUDA kernel returns:
            #   - cnt: [B, H, num_blocks] - directly the counts
            #   - offset: [B * H * num_blocks + 1] - flattened exclusive prefix sum (starts with 0)
            #   - idx: [total_blocks] - compact indices
            cuda_linear_k_block_sparse_mask = LinearBlockSparseTensorsTorch(
                mask_block_cnt=cuda_k_mask_cnt.flatten(),
                mask_block_offset=cuda_k_mask_offset,
                mask_block_idx=cuda_k_mask_idx,
                full_block_cnt=cuda_k_full_cnt.flatten(),
                full_block_offset=cuda_k_full_offset,
                full_block_idx=cuda_k_full_idx,
            )
        with torch.cuda.nvtx.range("create_k2q_csr_sparse_from_func"):
            # K2Q (Backward): fix kv_block, loop q_blocks
            (cuda_q_mask_cnt, cuda_q_mask_offset, cuda_q_mask_idx,
            cuda_q_full_cnt, cuda_q_full_offset, cuda_q_full_idx) = \
                create_block_mask_cuda.create_k2q_csr_sparse_from_func(
                    arbitrary_func,
                    seqlen_q,
                    seqlen_k,
                    Q_BLOCK_SIZE=tile_m,
                    KV_BLOCK_SIZE=tile_n
                )
            
            cuda_linear_q_block_sparse_mask = LinearBlockSparseTensorsTorch(
                mask_block_cnt=cuda_q_mask_cnt.flatten(),
                mask_block_offset=cuda_q_mask_offset,
                mask_block_idx=cuda_q_mask_idx,
                full_block_cnt=cuda_q_full_cnt.flatten(),
                full_block_offset=cuda_q_full_offset,
                full_block_idx=cuda_q_full_idx,
            )

        # =====================================================================
        # Verify CUDA kernel output matches PyTorch reference
        # =====================================================================
        def compare_linear_sparse_tensors(cuda_tensors, ref_tensors, name):
            """Compare CUDA kernel output with PyTorch reference."""
            all_match = True
            
            # Compare mask_block_cnt
            if not torch.equal(cuda_tensors.mask_block_cnt, ref_tensors.mask_block_cnt):
                print(f"  {name} mask_block_cnt MISMATCH!")
                print(f"    CUDA shape: {cuda_tensors.mask_block_cnt.shape}, ref shape: {ref_tensors.mask_block_cnt.shape}")
                print(f"    CUDA sum: {cuda_tensors.mask_block_cnt.sum().item()}, ref sum: {ref_tensors.mask_block_cnt.sum().item()}")
                all_match = False
            
            # Compare mask_block_offset
            if not torch.equal(cuda_tensors.mask_block_offset, ref_tensors.mask_block_offset):
                print(f"  {name} mask_block_offset MISMATCH!")
                print(f"    CUDA shape: {cuda_tensors.mask_block_offset.shape}, ref shape: {ref_tensors.mask_block_offset.shape}")
                all_match = False
            
            # Compare mask_block_idx (values should match, order may differ within each block)
            if cuda_tensors.mask_block_idx.shape != ref_tensors.mask_block_idx.shape:
                print(f"  {name} mask_block_idx shape MISMATCH!")
                print(f"    CUDA: {cuda_tensors.mask_block_idx.shape}, ref: {ref_tensors.mask_block_idx.shape}")
                all_match = False
            elif not torch.equal(cuda_tensors.mask_block_idx, ref_tensors.mask_block_idx):
                # Check if values are the same but in different order
                if set(cuda_tensors.mask_block_idx.tolist()) != set(ref_tensors.mask_block_idx.tolist()):
                    print(f"  {name} mask_block_idx values MISMATCH!")
                    all_match = False
            
            # Compare full_block_cnt
            if cuda_tensors.full_block_cnt is not None and ref_tensors.full_block_cnt is not None:
                if not torch.equal(cuda_tensors.full_block_cnt, ref_tensors.full_block_cnt):
                    print(f"  {name} full_block_cnt MISMATCH!")
                    all_match = False
            
            # Compare full_block_offset
            if cuda_tensors.full_block_offset is not None and ref_tensors.full_block_offset is not None:
                if not torch.equal(cuda_tensors.full_block_offset, ref_tensors.full_block_offset):
                    print(f"  {name} full_block_offset MISMATCH!")
                    all_match = False
            
            # Compare full_block_idx
            if cuda_tensors.full_block_idx is not None and ref_tensors.full_block_idx is not None:
                if cuda_tensors.full_block_idx.shape != ref_tensors.full_block_idx.shape:
                    print(f"  {name} full_block_idx shape MISMATCH!")
                    all_match = False
                elif not torch.equal(cuda_tensors.full_block_idx, ref_tensors.full_block_idx):
                    if set(cuda_tensors.full_block_idx.tolist()) != set(ref_tensors.full_block_idx.tolist()):
                        print(f"  {name} full_block_idx values MISMATCH!")
                        all_match = False
            
            return all_match

        print("\n" + "=" * 60)
        print("Comparing CUDA kernel vs PyTorch reference:")
        print("=" * 60)
        
        k_match = compare_linear_sparse_tensors(cuda_linear_k_block_sparse_mask, ref_linear_k_block_sparse_mask, "Q2K")
        q_match = compare_linear_sparse_tensors(cuda_linear_q_block_sparse_mask, ref_linear_q_block_sparse_mask, "K2Q")
        
        if k_match and q_match:
            print("✓ All CUDA kernel outputs match PyTorch reference!")
        else:
            print("✗ Some outputs do not match!")
            
        # Use CUDA kernel output for the attention computation
        linear_k_block_sparse_mask = cuda_linear_k_block_sparse_mask
        linear_q_block_sparse_mask = cuda_linear_q_block_sparse_mask
    else:
        # Fallback to PyTorch reference
        linear_k_block_sparse_mask = ref_linear_k_block_sparse_mask
        linear_q_block_sparse_mask = ref_linear_q_block_sparse_mask
    
    out_cute, lse_cute = flash_attn_func(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        softmax_scale=softmax_scale,
        causal=causal,
        arbitrary=True,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=False,
        deterministic=deterministic,
        mask_mod=None,
        linear_k_block_sparse_tensors=linear_k_block_sparse_mask,
        linear_q_block_sparse_tensors=linear_q_block_sparse_mask,
        aux_tensors=aux_tensors_arg,
    )

    out_ref_fp32 = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=True)
    out_ref = compute_reference_arbitrary(tensors, arbitrary_func, up_cast=False)

    print(f"Output max diff: {(out_cute - out_ref_fp32).abs().max().item()}")
    print(f"Pytorch max diff: {(out_ref - out_ref_fp32).abs().max().item()}")

    # # Check for invalid values
    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()
    
    assert (out_cute - out_ref_fp32).abs().max().item() <= 2 * (out_ref - out_ref_fp32).abs().max().item()

    # print("out_cute is {}".format(out_cute))
    # print("out_ref_fp32 is {}".format(out_ref_fp32))

    dout = torch.rand_like(out_cute)

    dq, dk, dv = torch.autograd.grad(
        out_cute, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref_fp32, dk_ref_fp32, dv_ref_fp32) = torch.autograd.grad(
        out_ref_fp32, (tensors["q"], tensors["k"], tensors["v"]), dout
    )
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
        out_ref, (tensors["q"], tensors["k"], tensors["v"]), dout
    )

    print(f"dV max diff: {(dv - dv_ref_fp32).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_ref - dv_ref_fp32).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref_fp32).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_ref - dk_ref_fp32).abs().max().item()}")
    print(f"dQ max diff: {(dq - dq_ref_fp32).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_ref - dq_ref_fp32).abs().max().item()}")

    # diff = (dq - dq_ref).abs()
    # print(f"shape diff: {diff.shape}")  # b s h d
    # for bs in range(batch_size):
    #     for h in range(nheads):
    #         for s in range(seqlen_q):
    #             diff_i = diff[bs, s, h, 0] #只取当前dim的第一个值
    #             if diff_i.abs().max().item() > 1e-2:
    #                 out_i = dq[bs, s, h, 0]
    #                 out_ref_i = dq_ref[bs, s, h, 0]
    #                 print(f"=== out[bs, s, h, 0] = {out_i}, out_ref[bs, s, h, 0] = {out_ref_i}, diff[bs, s, h, 0] = {diff_i}, bs = {bs}, s = {s}, h = {h}")
    
    assert (dv - dv_ref_fp32).abs().max().item() <= 5 * (dv_ref - dv_ref_fp32).abs().max().item()
    assert (dk - dk_ref_fp32).abs().max().item() <= 5 * (dk_ref - dk_ref_fp32).abs().max().item()
    assert (dq - dq_ref_fp32).abs().max().item() <= 5 * (dq_ref - dq_ref_fp32).abs().max().item()


def test_arbitrary_mask_block_score_and_lse():
    """``max_score_out`` + ``block_lse_out`` simultaneously (dense bidirectional, SM100 only)."""
    if COMPUTE_CAPABILITY != 10:
        pytest.skip("max_score_out / block_lse_out is only implemented on SM 10.x cute forward")

    seqlen_q, seqlen_k = 256, 256
    nheads = 4
    headdim = 128
    dtype = torch.bfloat16
    batch_size = 1
    nheads_kv = nheads

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim, dtype
    )
    headdim = tensors["q"].shape[3]
    softmax_scale = 1.0 / math.sqrt(headdim)

    num_k_chunks = (seqlen_k + 127) // 128
    max_score_out = torch.empty(
        batch_size, nheads, seqlen_q, num_k_chunks,
        dtype=torch.float32,
        device="cuda",
    )
    block_lse_out = torch.empty(
        batch_size, nheads, seqlen_q, num_k_chunks,
        dtype=torch.float32,
        device="cuda",
    )

    out, lse = _flash_attn_fwd(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=False,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=False,
        mask_mod=None,
        block_sparse_tensors=None,
        aux_tensors=None,
        max_score_out=max_score_out,
        block_lse_out=block_lse_out,
        k_sparse_block_size=128,
    )

    assert out.shape == tensors["out"].shape
    assert torch.isfinite(out).all()

    ref_max = compute_reference_block_max_scores(tensors, arbitrary_func=None)
    assert torch.isfinite(max_score_out).all()
    diff_ms = (max_score_out - ref_max).abs()
    print(f"max_score_out diff max: {diff_ms.max().item()}, avg: {diff_ms.mean().item()}")
    assert torch.allclose(
        max_score_out, ref_max, rtol=0.06, atol=0.2,
    ), f"max_score mismatch max diff {diff_ms.max().item()}"

    ref_lse = compute_reference_block_lse(tensors, softmax_scale, arbitrary_func=None)
    assert torch.isfinite(block_lse_out).all()
    diff_lse = (block_lse_out - ref_lse).abs()
    print(f"block_lse_out diff max: {diff_lse.max().item()}, avg: {diff_lse.mean().item()}")
    assert torch.allclose(
        block_lse_out, ref_lse, rtol=0.06, atol=0.2,
    ), f"block_lse mismatch max diff {diff_lse.max().item()}"

    _ = lse


def test_arbitrary_mask_block_score_and_lse_sparse():
    """``max_score_out`` + ``block_lse_out`` with block sparsity + arbitrary mask (SM100)."""
    if COMPUTE_CAPABILITY != 10:
        pytest.skip("max_score_out / block_lse_out block-sparse path is only exercised on SM 10.x")

    torch.manual_seed(0)

    seqlen_q, seqlen_k = 256, 256
    tile_m, tile_n = 128, 128
    sparse_tile_m = 2 * tile_m
    nheads = 4
    nheads_kv = nheads
    headdim = 128
    dtype = torch.bfloat16
    batch_size = 1

    _, mask_mod_flex = get_mask_pair("arbitrary")
    arbitrary_func = random_arbitrary_func_tensor(
        1, 1, 3, seqlen_q, seqlen_k, device="cuda"
    )
    frozen_af = arbitrary_func

    def mask_mod_wrap(b, h, q_idx, kv_idx, f=frozen_af):
        return mask_mod_flex(b, h, q_idx, kv_idx, f)

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim, dtype
    )
    headdim = tensors["q"].shape[3]
    softmax_scale = 1.0 / math.sqrt(headdim)

    bm = create_block_mask(
        mask_mod_wrap,
        1,
        1,
        seqlen_q,
        seqlen_k,
        device="cuda",
        BLOCK_SIZE=(sparse_tile_m, tile_n),
    )
    _, _, k_mask_cnt, k_mask_idx, k_full_cnt, k_full_idx, *_ = bm.as_tuple()
    k_bsm = BlockSparseTensorsTorch(
        mask_block_cnt=k_mask_cnt,
        mask_block_idx=k_mask_idx,
        full_block_cnt=k_full_cnt,
        full_block_idx=k_full_idx,
    )
    linear_k = bhqk_to_linear_sparse_tensors(k_bsm)

    num_k_chunks = (seqlen_k + 127) // 128
    max_score_out = torch.empty(
        batch_size, nheads, seqlen_q, num_k_chunks,
        dtype=torch.float32,
        device="cuda",
    )
    block_lse_out = torch.empty(
        batch_size, nheads, seqlen_q, num_k_chunks,
        dtype=torch.float32,
        device="cuda",
    )

    out, lse = _flash_attn_fwd(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        softmax_scale=softmax_scale,
        causal=False,
        arbitrary=True,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        num_splits=1,
        pack_gqa=False,
        mask_mod=None,
        block_sparse_tensors=linear_k,
        aux_tensors=[arbitrary_func],
        max_score_out=max_score_out,
        block_lse_out=block_lse_out,
        k_sparse_block_size=128,
    )

    assert out.shape == tensors["out"].shape
    assert torch.isfinite(out).all()

    # --- max_score check (per-doc block scoring) ---
    ref_max = compute_reference_per_doc_block_max_scores(
        tensors, arbitrary_func, linear_k=linear_k, q_super_block=sparse_tile_m,
    )
    diff_ms = (max_score_out - ref_max).abs()
    print(f"max_score_out sparse diff max: {diff_ms.max().item()}, avg: {diff_ms.mean().item()}")
    assert torch.allclose(
        max_score_out, ref_max, rtol=0.06, atol=0.2,
    ), f"sparse max_score mismatch max diff {diff_ms.max().item()}"

    # --- block_lse check (per-doc block scoring) ---
    ref_lse = compute_reference_per_doc_block_lse(
        tensors, softmax_scale, arbitrary_func,
        linear_k=linear_k, q_super_block=sparse_tile_m,
    )
    diff_lse = (block_lse_out - ref_lse).abs()
    finite_mask = torch.isfinite(ref_lse)
    diff_finite = diff_lse[finite_mask]
    print(f"block_lse_out sparse diff max: {diff_finite.max().item()}, avg: {diff_finite.mean().item()}")
    assert torch.allclose(
        block_lse_out[finite_mask], ref_lse[finite_mask], rtol=0.06, atol=0.2,
    ), f"sparse block_lse mismatch max diff {diff_finite.max().item()}"
    _ = lse


def test_arbitrary_mask(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, tile_m, tile_n, load_tensor=False, deterministic=False
):
    """Test arbitrary mask
    """
    if COMPUTE_CAPABILITY == 10 and (tile_m, tile_n) != (128, 128):
        pytest.skip("TODO: Non-128x128 tiles currently not supported on SM 10.0. due to TMEM")

    _run_mask_test(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        nheads=nheads,
        kv_mode=kv_mode,
        headdim=headdim,
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        use_block_sparsity=use_block_sparsity,
        load_tensor=load_tensor,
        deterministic=deterministic,
    )


if __name__ == "__main__":
    # Test non-deterministic mode
    print("=" * 60)
    print("Testing non-deterministic mode (deterministic=False)")
    print("=" * 60)
    test_arbitrary_mask(
        seqlen_q=1024,
        seqlen_k=1024,
        nheads=4,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        tile_m=128,
        tile_n=128,
        load_tensor=False,
        deterministic=False,
    )
    
    # Test deterministic mode
    print("\n" + "=" * 60)
    print("Testing deterministic mode (deterministic=True)")
    print("=" * 60)
    test_arbitrary_mask(
        seqlen_q=1024,
        seqlen_k=1024,
        nheads=4,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=True,
        tile_m=128,
        tile_n=128,
        load_tensor=False,
        deterministic=True,
    )
