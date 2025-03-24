#!POPCORN leaderboard vectorsum

# This is a submission template for popcorn leaderboard 'vectorsum'.
# Your task is as follows:
# > Implement a vector sum reduction kernel. This kernel computes the sum of all elements in the input tensor.

# > 

# > Input: A tensor of shape `(N,)` with values from a normal distribution with mean 0 and variance 1.

# > Output: A scalar value equal to the sum of all elements in the input tensor.

# The deadline for this leaderboard is 2025-04-30 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t
import triton 
import triton.language as tl


# @triton.jit
# def kernel(data_ptr, output_ptr, n):
#     pass

def custom_kernel(data: input_t) -> output_t:
    if data.device != 'cuda':
        data = data.cuda()

