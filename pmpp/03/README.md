## Multidimensional grid


Each block is labeled with `(blockIdx.y, blockIdx.x)`. For example, block (1,0) has
`blockIdx.y = 1` and `blockIdx.x = 0`.


Example: A block of 4x2x2 threads. Thread (1,0,2) has `threadIdx.z = 1`,
`threadIdx.y = 0`, and `threadIdx.x = 2`.

The order is backward compared to Numpy notation (row, col, channel). Cuda: (channel, col, row)

Ordering of the block and
thread labels is such that highest dimension comes first `(z, y, x)`. reverse of that used in the C statements for setting configuration parameters, in which the lowest dimension comes first `(x, y, z)`




`Divergence` could happens if branch condition involves threadId
 * Threads in a warp takes different path 


