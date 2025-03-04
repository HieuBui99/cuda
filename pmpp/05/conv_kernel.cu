#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_SIZE 16

static __constant__ float F[5][5];


__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void basic_conv1d(float* inp, float* kernel, float* out, int width, int kernel_width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    int inp_start_index = i - kernel_width / 2;
    for (int i = 0; i < kernel_width; i++) {
        if (inp_start_index + i >= 0 && inp_start_index + i < width) {
            sum += inp[inp_start_index + i] * kernel[i];
        }
    }
    out[i] = sum;
}

__global__ void basic_conv2d(float* inp, float* kernel, float* out, int height, int width, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int inp_row_start_idx = row - kernel_size / 2;
    int inp_col_start_idx = col - kernel_size / 2;
    
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            if (inp_row_start_idx + i >= 0 && inp_row_start_idx + i < height && inp_col_start_idx + j >= 0 && inp_col_start_idx + j < width) {
                sum += inp[(inp_row_start_idx+i)*width + (inp_col_start_idx+j)] * kernel[i*kernel_size + j];
            }
        }
    }
    out[row * width + col] = sum;
}

__global__ void conv2d_with_constant_mem(float* inp, float* out, int height, int width, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int inp_row_start_idx = row - kernel_size / 2;
    int inp_col_start_idx = col - kernel_size / 2;
    
    float sum = 0.0f;
    
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            if (inp_row_start_idx + i >= 0 && inp_row_start_idx + i < height && inp_col_start_idx + j >= 0 && inp_col_start_idx + j < width) {
                sum += inp[(inp_row_start_idx+i)*width + (inp_col_start_idx+j)] * F[i][j];
            }
        }
    }
    out[row * width + col] = sum;
}

__global__ void conv2d_tiled_with_constant_mem(float* inp, float* out, int height, int width, int kernel_size) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        tile[threadIdx.y][threadIdx.x] = inp[row * width + col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int inp_row_start_idx = row - kernel_size / 2;
    int inp_col_start_idx = col - kernel_size / 2;
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                if (threadIdx.y - kernel_size / 2 + i >= 0 && 
                    threadIdx.y - kernel_size / 2 + i < TILE_SIZE && 
                    threadIdx.x - kernel_size / 2 + j >= 0 && 
                    threadIdx.x - kernel_size / 2 + j < TILE_SIZE) {
                        sum += F[i][j] * tile[threadIdx.y - kernel_size / 2 + i][threadIdx.x - kernel_size / 2 + j];
                }
                else {
                    if (inp_row_start_idx + i >= 0 && inp_row_start_idx + i < height && inp_col_start_idx + j >= 0 && inp_col_start_idx + j < width) {
                        sum += inp[(inp_row_start_idx+i)*width + (inp_col_start_idx+j)] * F[i][j];
                    }
                }
            }
        }
        out[row * width + col] = sum;

    }

}


torch::Tensor conv1d_cuda(torch::Tensor inp, torch::Tensor kernel, int width, int kernel_width) {
    int num_blocks = cdiv(width, TILE_SIZE);

    auto out = torch::empty({width}, inp.options());
    basic_conv1d<<<num_blocks, TILE_SIZE>>>(inp.data_ptr<float>(), kernel.data_ptr<float>(), out.data_ptr<float>(), width, kernel_width);
    return out;
}


torch::Tensor conv2d_cuda(torch::Tensor inp, torch::Tensor kernel, int height, int width, int kernel_size) {
    dim3 num_blocks(cdiv(width, TILE_SIZE), cdiv(height, TILE_SIZE));
    dim3 block_size(TILE_SIZE, TILE_SIZE);

    auto out = torch::empty({height, width}, inp.options());
    basic_conv2d<<<num_blocks, block_size>>>(inp.data_ptr<float>(), kernel.data_ptr<float>(), out.data_ptr<float>(), height, width, kernel_size);
    // cudaMemcpyToSymbol(F, kernel.data_ptr<float>(), kernel_size * kernel_size * sizeof(float));
    
    // conv2d_with_constant_mem<<<num_blocks, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), height, width, kernel_size);
    return out;
}

torch::Tensor conv2d_with_constant_mem(torch::Tensor inp, torch::Tensor kernel, int height, int width, int kernel_size) {
    dim3 num_blocks(cdiv(width, TILE_SIZE), cdiv(height, TILE_SIZE));
    dim3 block_size(TILE_SIZE, TILE_SIZE);

    auto out = torch::empty({height, width}, inp.options());
    cudaMemcpyToSymbol(F, kernel.data_ptr<float>(), kernel_size * kernel_size * sizeof(float));
    
    conv2d_tiled_with_constant_mem<<<num_blocks, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), height, width, kernel_size);
    return out;
}
