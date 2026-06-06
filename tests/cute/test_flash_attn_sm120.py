# Copyright (c) 2026, FlashAttention contributors.

import math

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd


def _is_sm120_family():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


@pytest.mark.skipif(not _is_sm120_family(), reason="requires an SM120-family GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_forward_matches_sdpa(dtype, head_dim, causal):
    torch.manual_seed(0)
    q = torch.randn(1, 128, 4, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out, _ = _flash_attn_fwd(q, k, v, causal=causal)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=1.0 / math.sqrt(head_dim),
        is_causal=causal,
    ).transpose(1, 2)

    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not _is_sm120_family(), reason="requires an SM120-family GPU")
def test_sm120_arbitrary_mask_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    prefix_lengths = torch.arange(1, 129, device="cuda", dtype=torch.int32).reshape(1, 1, 1, 128)

    out, _ = _flash_attn_fwd(q, k, v, arbitrary=True, aux_tensors=[prefix_lengths])
    mask = torch.arange(128, device="cuda").reshape(
        1, 1, 1, 128
    ) < prefix_lengths.squeeze(2).unsqueeze(-1)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)

    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
