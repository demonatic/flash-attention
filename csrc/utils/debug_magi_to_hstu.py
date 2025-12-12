#!/usr/bin/env python3
"""
Test script for magi_to_hstu CUDA kernel.
Ported from the original C++ main() test function.

Test cases from https://sandai-org.github.io/MagiAttention/blog/
"""

import torch
import magi_to_hstu_cuda


def decode_func_out(func_out, q_idx):
    """
    Decode func_out to get valid k intervals for a given q position.
    
    Function encoding rule:
        interval 0: [0, F[0])
        interval 1: [F[1], F[2])
        interval 2: [F[3], F[4])
        ...
    """
    intervals = []
    n_max_func = func_out.shape[0]
    
    f0 = func_out[0, q_idx].item()
    if f0 == -1:
        return intervals
    
    if f0 > 0:
        intervals.append((0, f0))
    
    for i in range(1, n_max_func, 2):
        start = func_out[i, q_idx].item()
        end_idx = i + 1
        if end_idx >= n_max_func:
            break
        end = func_out[end_idx, q_idx].item()
        if start == -1 or end == -1:
            break
        intervals.append((start, end))
    
    return intervals


def reconstruct_mask(func_out, seqlen_q, seqlen_k):
    """Reconstruct attention mask from func_out."""
    mask = torch.zeros(seqlen_q, seqlen_k, dtype=torch.bool)
    
    for q in range(seqlen_q):
        intervals = decode_func_out(func_out, q)
        for start, end in intervals:
            mask[q, start:end] = True
    
    return mask


def print_mask(mask):
    """Print attention mask in a readable format."""
    seqlen_q, seqlen_k = mask.shape
    
    # Header
    print("     ", end="")
    for k in range(seqlen_k):
        print(f"k={k:<2} ", end="")
    print()
    
    # Rows
    for q in range(seqlen_q):
        print(f"q={q:<2} ", end="")
        for k in range(seqlen_k):
            print(f"  {'1' if mask[q, k] else '0'}  ", end="")
        print()


def debug_mask():
    q_ranges = torch.load("q_ranges.pt")
    k_ranges = torch.load("k_ranges.pt")
    mask_types = torch.load("attn_type_map.pt")
    # ref_func_out = torch.load("hstu_func.pt")
    seqlen_q = 14336
    seqlen_k = 14336
    n_max_func = 8192
    
    print(f"\n=== Attention Slice Configuration ===")
    print(f"{seqlen_q=}, {seqlen_k=}, {n_max_func=}\n")
    print(f"{q_ranges=} | {q_ranges.shape=}\n")
    print(f"{k_ranges=} | {k_ranges.shape=}\n")
    print(f"{mask_types=} | {mask_types.shape=}\n")
    
    print("\n=== Function Output ===")
    func_out = magi_to_hstu_cuda.magi_to_hstu(
        q_ranges, k_ranges, mask_types, seqlen_q, seqlen_k, n_max_func
    ).unsqueeze(0).unsqueeze(0)
    print(f"{func_out=}\n")
    # torch.testing.assert_close(func_out, ref_func_out)
    
    # print("\n=== Attention Mask (reconstructed from func_out) ===")
    # mask = reconstruct_mask(func_out, seqlen_q, seqlen_k)
    # print_mask(mask)
    
    # return func_out


if __name__ == "__main__":
    print("Debuging magi_to_hstu CUDA kernel\n")
    
    debug_mask()

