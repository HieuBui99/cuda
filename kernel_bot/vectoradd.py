#!POPCORN leaderboard vectoradd

# This is a submission template for popcorn leaderboard 'vectoradd'.
# Your task is as follows:
# > Implement a float16 vector addition kernel.

# > 

# > Input: tuple(torch.Tensor, torch.Tensor) with tensors of shape (N, N) and type torch.float16. These tensors are from

# > a normal distribution with mean 0 and variance 1.

# > Output: torch.Tensor of shape (N, N) and type torch.float16

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
import triton
import torch
import triton.language as tl
from torch.utils.cpp_extension import load_inline

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, num_warps: tl.constexpr):
    pid = tl.program_id(0)

    offset = pid * BLOCK_SIZE
    idx = tl.arange(0, BLOCK_SIZE) + offset
    mask = idx < n_elements

    x = tl.load(x_ptr + idx, mask=mask)
    y = tl.load(y_ptr + idx, mask=mask)
    output = x + y

    tl.store(output_ptr + idx, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    shape = x.shape

    x = x.view(-1)
    y = y.view(-1)

    output = torch.empty_like(x)
    n_elements = output.numel()

    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    if BLOCK_SIZE > 65536:
        BLOCK_SIZE = 65536
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.

    output = output.view(shape)
    return output


def custom_kernel(data: input_t) -> output_t:
    x, y = data
    shape = x.shape

    x = x.view(-1).to(torch.float8_e4m3fn)
    y = y.view(-1).to(torch.float8_e4m3fn)

    output = torch.empty_like(x)
    n_elements = output.numel()

    BLOCK_SIZE = triton.next_power_of_2(n_elements)
    if BLOCK_SIZE > 65536:
        BLOCK_SIZE = 65536
    
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.

    output = output.view(shape)
    return output
    

