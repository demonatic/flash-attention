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
from einops import rearrange

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.mask_definitions import (
    get_mask_pair,
    STATIC_MASKS,
    random_arbitrary_func_tensor,
)
from flash_attn.cute.testing import attention_ref
COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


@pytest.fixture(autouse=True)
def reset_torch_state():
    """Reset torch dynamo/compile state between tests to avoid state pollution."""
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    yield

    torch._dynamo.reset()
    torch.cuda.empty_cache()

def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    lengths = torch.randint(1, seqlen_q + 1, (batch_size,))
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    total_q = cu_seqlens_q[-1]
    total_k = total_q
    q = torch.empty(total_q, nheads, headdim, device=device, dtype=dtype).uniform_(-1, 1)
    k = torch.empty(
        total_k, nheads_kv, headdim, device=device, dtype=dtype
    ).uniform_(-1, 1)
    v = torch.empty(
        total_k, nheads_kv, headdim_v, device=device, dtype=dtype
    ).uniform_(-1, 1)
    out = torch.empty(
        total_q, nheads, headdim_v, device=device, dtype=dtype
    )
    lse = torch.empty(nheads, total_q, device=device, dtype=torch.float32)

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "out": out.contiguous(),
        "lse": lse.contiguous(),
        "cu_seqlens_q": cu_seqlens_q.contiguous(),
        "cu_seqlens_k": cu_seqlens_q.contiguous(),
    }

def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(
            torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i])
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, : (cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)


def compute_reference_arbitrary(tensors, max_seqlen_q, max_seqlen_k, mask_mod_flex, block_size: Optional[tuple[int, int]] = None):
    """Compute reference of arbitrary mask"""
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    cu_seqlens_q = tensors["cu_seqlens_q"]
    cu_seqlens_k = tensors["cu_seqlens_k"]
    batch_size = cu_seqlens_q.shape[0] - 1
    nheads = q.shape[1]
    headdim = q.shape[2]

    padded_q = pad_input(q, cu_seqlens_q, batch_size, max_seqlen_q)
    padded_k = pad_input(k, cu_seqlens_k, batch_size, max_seqlen_k)
    padded_v = pad_input(v, cu_seqlens_k, batch_size, max_seqlen_k)

    padded_q = padded_q.view(batch_size, max_seqlen_q, nheads, headdim)
    padded_k = padded_k.view(batch_size, max_seqlen_k, nheads, headdim)
    padded_v = padded_v.view(batch_size, max_seqlen_k, nheads, headdim)

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q,
        padded_k,
    )

    return s

    scale = 1.0 / math.sqrt(headdim)

    # Handle identity (no masking) case
    if mask_mod_flex is None:
        out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
        return out_ref.transpose(1, 2).contiguous()

    block_mask_kwargs = {}
    if block_size is not None:
        block_mask_kwargs["BLOCK_SIZE"] = block_size

    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device=q.device,
        **block_mask_kwargs,
    )
    out_ref = flex_attention(q, k, v, block_mask=block_mask, scale=scale)
    return out_ref.transpose(1, 2).contiguous()


SEQLEN_PAIRS_COMPREHENSIVE = [
    (1, 1),
    (64, 128),
    (128, 192),
    (256, 256),
    (239, 1),
    (799, 3),
    (113, 203),
    (113, 128),
    (128, 217),
    (113, 211),
    (108, 256),
    (256, 512),
    (384, 256),
    (640, 128),
    (512, 256),
    (1024, 1024),
    (1023, 1024),
    (1024, 1023),
    (4096, 4096),
    (4224, 4224),
]

SEQLEN_PAIRS_SMOKE = [
    (128, 128),
    (256, 256),
    (113, 203),
    (1024, 1024),
    (128, 8192)
]


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
):
    torch.manual_seed(42)

    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 8
    headdim_v = headdim

    mask_mod_cute, mask_mod_flex = get_mask_pair("arbitrary")
    arbitrary_func = random_arbitrary_func_tensor(1, batch_size, 1, seqlen_q, seqlen_k, device="cuda")
    original_flex_mask = mask_mod_flex

    def mask_mod_flex(b, h, q_idx, kv_idx, arbitrary_func=arbitrary_func):
        return original_flex_mask(b, h, q_idx, kv_idx, arbitrary_func)

    aux_tensors_arg = [arbitrary_func]
    causal = False

    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    # Compute block sparsity for mask_mod
    if COMPUTE_CAPABILITY == 10:
        sparse_tile_m = 2 * tile_m
    else:
        sparse_tile_m = tile_m

    softmax_scale = 1.0 / math.sqrt(headdim)

    out_tuple = _flash_attn_fwd(
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        out=tensors["out"],
        lse=tensors["lse"],
        cu_seqlens_q=tensors["cu_seqlens_q"],
        cu_seqlens_k=tensors["cu_seqlens_k"],
        seqused_q=None,
        seqused_k=None,
        page_table=None,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=None,
        window_size_left=None,
        window_size_right=None,
        learnable_sink=None,
        m_block_size=tile_m,
        n_block_size=tile_n,
        num_threads=384,
        pack_gqa=False,
        _compute_capability=None,
        score_mod=None,
        mask_mod=mask_mod_cute,
        block_sparse_tensors=None,
        return_lse=True,
        aux_tensors=aux_tensors_arg,
    )

    out_cute = out_tuple[0]
    q_padded = pad_input(tensors["q"], tensors["cu_seqlens_q"], batch_size, seqlen_q)
    k_padded = pad_input(tensors["k"], tensors["cu_seqlens_k"], batch_size, seqlen_k)
    v_padded = pad_input(tensors["v"], tensors["cu_seqlens_k"], batch_size, seqlen_k)

    block_size = (tile_m, tile_n)
    out_ref_fp32, _ = attention_ref(
      q=q_padded,
      k=k_padded,
      v=v_padded,
      query_padding_mask=None,
      key_padding_mask=None,
      key_leftpad=None,
      attn_bias=None,
      dropout_p=0.0,
      dropout_mask=None,
      causal=True,
      qv=None,
      q_descale=None,
      k_descale=None,
      v_descale=None,
      window_size=(None, None),
      attention_chunk=0,
      sink_token_length=0,
      learnable_sink=None,
      softcap=0.0,
      upcast=True,
      reorder_ops=False,
      intermediate_dtype=None,
    )
    out_ref, _ = attention_ref(
      q=q_padded,
      k=k_padded,
      v=v_padded,
      query_padding_mask=None,
      key_padding_mask=None,
      key_leftpad=None,
      attn_bias=None,
      dropout_p=0.0,
      dropout_mask=None,
      causal=True,
      qv=None,
      q_descale=None,
      k_descale=None,
      v_descale=None,
      window_size=(None, None),
      attention_chunk=0,
      sink_token_length=0,
      learnable_sink=None,
      softcap=0.0,
      upcast=False,
      reorder_ops=False,
      intermediate_dtype=None,
    )
    out_ref_fp32 = unpad_input(out_ref_fp32, tensors["cu_seqlens_q"])
    out_ref = unpad_input(out_ref, tensors["cu_seqlens_q"])

    # Check for invalid values
    assert out_cute.shape == out_ref_fp32.shape == out_ref.shape
    assert not torch.isnan(out_cute).any()
    assert not torch.isnan(out_ref_fp32).any()
    assert torch.isfinite(out_cute).all()
    assert torch.isfinite(out_ref_fp32).all()

    # Compute numerical tolerance (matching flash attention tests)
    fwd_atol = 2 * (out_ref_fp32 + 0.3 - 0.3 - out_ref_fp32).abs().max().item()
    rtol = 2

    ref_error = (out_ref - out_ref_fp32).abs().max().item()
    cute_error = (out_cute - out_ref_fp32).abs().max().item()

    mask_desc = f"mask_mod=arbitrary_causal"

    print(
        f"\n{mask_desc} @ Q={seqlen_q}, K={seqlen_k}, H={nheads}/{nheads_kv} ({kv_mode}), "
        f"D={headdim}, M={tile_m}, N={tile_n}"
    )
    print("  Reference implementation: FlexAttention")
    print(f"  Reference vs FP32: {ref_error:.2e}")
    print(f"  Kernel vs FP32: {cute_error:.2e}")
    print(f"  Tolerance: rtol={rtol} * {ref_error:.2e} + {fwd_atol:.2e}")
    print(f"  Error ratio: {cute_error / max(ref_error, 1e-10):.2f}")

    # Debug: show some sample values if error is large
    if cute_error > 1e-2:
        print(f"  DEBUG: Sample kernel output: {out_cute[0, 0, :5]}")
        print(f"  DEBUG: Sample reference output: {out_ref_fp32[0, 0, :5]}")
        print(f"  DEBUG: Max diff location: {(out_cute - out_ref_fp32).abs().argmax()}")
        max_diff_idx = (out_cute - out_ref_fp32).abs().argmax()
        max_diff_coords = torch.unravel_index(max_diff_idx, out_cute.shape)
        print(f"  DEBUG: Max diff at coords: {max_diff_coords}")
        print(f"  DEBUG: Kernel value: {out_cute[max_diff_coords]:.6f}")
        print(f"  DEBUG: Reference value: {out_ref_fp32[max_diff_coords]:.6f}")

    # Use the same assertion logic as FlashAttention tests
    assert cute_error <= rtol * ref_error + fwd_atol, (
        f"Kernel error {cute_error:.2e} exceeds {rtol}x PyTorch error {ref_error:.2e} + {fwd_atol:.2e}"
    )


@pytest.mark.parametrize("seqlen_q,seqlen_k", SEQLEN_PAIRS_SMOKE)
@pytest.mark.parametrize("nheads", [16])
@pytest.mark.parametrize("kv_mode", ["mha"])
@pytest.mark.parametrize("headdim", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_block_sparsity", [True, False])
@pytest.mark.parametrize("tile_m,tile_n", [(128, 128), (128, 112), (64, 128)])
def test_arbitrary_mask(
    seqlen_q, seqlen_k, nheads, kv_mode, headdim, dtype, use_block_sparsity, tile_m, tile_n
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
    )


if __name__ == "__main__":
    test_arbitrary_mask(
        seqlen_q=8192,
        seqlen_k=8192,
        nheads=8,
        kv_mode="mha",
        headdim=128,
        dtype=torch.bfloat16,
        use_block_sparsity=False,
        tile_m=128,
        tile_n=128,
    )