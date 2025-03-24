#!POPCORN leaderboard grayscale

# This is a submission template for popcorn leaderboard 'grayscale'.
# Your task is as follows:
# > Implement an RGB to grayscale conversion kernel that matches the reference implementation.

# > The kernel should convert square RGB images with even sizes to grayscale using the standard coefficients:

# > Y = 0.2989 R + 0.5870 G + 0.1140 B

# > 

# > Input: RGB tensor of shape (H, W, 3) with values in [0, 1]

# > Output: Grayscale tensor of shape (H, W) with values in [0, 1]

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
import triton
import torch
import triton.language as tl

@triton.jit
def kernel(data_ptr, output_ptr, stride_c, stride_row, stride_col, bs0: tl.constexpr, bs1: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    



def custom_kernel(data: input_t) -> output_t:
    if data.device != 'cuda':
        data = data.cuda()

    #data: (H, W, 3)
    h, w, c = data.shape
    data = data.view(c, h, w)
    output = torch.empty((h, w), dtype=data.dtype, device=data.device)

    # Call the kernel
    grid = lambda args: (tl.cdiv(h, args[0]), tl.cdiv(w, args[1]))  # noqa: E731
    kernel[grid](data, output, data.stride(0), data.stride(1), data.stride(2))

