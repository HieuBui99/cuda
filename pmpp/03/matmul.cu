#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void mysgemm_v2(int m, int n, int k, const float *A, const float *B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row  < m ) {
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++) {
                sum += A[row * k + j] * B[j * n + i];
            }
            C[row * n + i] = sum;
        }
    }

}

__global__ void mysgemm_v3(int m, int n, int k, const float *A, const float *B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row  < m ) {
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++) {
                sum += A[row * k + j] * B[j * n + i];
            }
            C[row * n + i] = sum;
        }
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

    // dim3 dimGrid(cdiv(w, 16), cdiv(h, 16));
    // dim3 dimBlock(16, 16);
    dim3 dimGrid(w, cdiv(h, 32), 1);
    dim3 dimBlock(1, 32, 1);
    mysgemm_v3<<<cdiv(h, 256), 256>>>(h, w, k, a.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

