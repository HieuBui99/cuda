#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void tileMatmulKernel(int m, int n, int k, const float *A, const float *B, float* C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0;

    for (int ph = 0; ph < cdiv(k, TILE_SIZE); ph++) {
        int idx = ph * TILE_SIZE;

        if (row < m && idx+tx < k){
            As[ty][tx] = A[row*k + idx + tx];
        }
        else {
            As[ty][tx] = 0.0f;
        }
        if (col < n && idx+ty < k){
            Bs[ty][tx] = B[(idx+ty)*n + col];
        }
        else {
            Bs[ty][tx] = 0.0f;
        }   
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n){
        C[row * n + col] = sum;
    }
}


torch::Tensor torchMatmul(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    int h = a.size(0);
    int w = b.size(1);
    int k = a.size(1);
    TORCH_CHECK(k==b.size(0), "Size mismatch!");

    auto output = torch::empty({h, w}, a.options());

    dim3 dimGrid(cdiv(w, TILE_SIZE), cdiv(h, TILE_SIZE));
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    // dim3 dimGrid(w, cdiv(h, 32), 1);
    // dim3 dimBlock(1, 32, 1);
    tileMatmulKernel<<<dimGrid, dimBlock>>>(h, w, k, a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

