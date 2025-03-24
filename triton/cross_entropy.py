import torch
from torch import nn
import triton
import triton.language as tl

MAX_FUSED_SIZE : int = 65536
def calculate_settings(n : int):
    BLOCK_SIZE : int = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr, log_sum_exp_ptr,
                         row_stride, num_class, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)

    logits_ptr += row_idx * row_stride
    labels_ptr += row_idx 
    loss_ptr += row_idx
    log_sum_exp_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_class
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other = -float("inf"))
    labels = tl.load(labels_ptr)
    # log_sum_exp = tl.logsumexp(logits, axis=0)
    
    row_max = tl.max(logits, axis=0)
    row_sum = tl.sum(tl.exp(logits - row_max), axis=0)

    x = tl.load(logits_ptr + labels)
    x = tl.exp(x - row_max) / row_sum
    loss = - 1.0 * tl.log(x)
    tl.store(loss_ptr, loss)


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor):
    """"
        logits: batch_size x num_classes
        labels: batch_size
    """
    b, d = logits.shape
    BLOCK_SIZE, num_warps = calculate_settings(d)
    loss = torch.empty((b,), dtype=logits.dtype, device=logits.device)
    log_sum_exp = torch.empty((b,), dtype=logits.dtype, device=logits.device)


    grid = (b, )

    cross_entropy_kernel[grid](logits, labels, loss, log_sum_exp,
                               logits.stride(0), num_class=d, BLOCK_SIZE=BLOCK_SIZE)
    return loss