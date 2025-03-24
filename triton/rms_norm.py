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
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride : tl.constexpr,
    n_cols     : tl.constexpr,
    eps        : tl.constexpr,
    BLOCK_SIZE : tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    # Y = tl.advance(Y, row_idx * Y_row_stride)
    # X = tl.advance(X, row_idx * X_row_stride)
    # r = tl.advance(r, row_idx * r_row_stride)


    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)#.to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype) # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)


def rms_norm_forward(X: torch.Tensor, W: torch.Tensor, eps: float):
    shape = X.shape
    dim : int = shape[-1]
    X = X.view(-1, dim)

    n_rows, n_cols = X.shape

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    device = X.device

    Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = device) 
    r = torch.empty(n_rows, dtype = torch.float32, device = device) #inverse std

    fx = _rms_layernorm_forward
    with torch.cuda.device(device):
        fx[(n_rows,)](
            Y, Y.stride(0),
            X, X.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            n_cols, eps,
            BLOCK_SIZE = BLOCK_SIZE,
            num_warps  = num_warps,
        )

    return Y.view(*shape)