#include <cmath>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE_ROW 512
#define BLOCK_SIZE_COL 1024

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}


__global__ void naive_softmax_kernel(float* input, float* output, int num_classes) {
    // Each thread calculate 1 row of the
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_val = -INFINITY;
    for (int i = 0; i < num_classes; i++) {
        max_val = fmaxf(max_val, input[idx * num_classes + i]);
    }

    float sum_exp = 0;
    for (int i = 0; i < num_classes; i++) {
        sum_exp = sum_exp + expf(input[idx * num_classes + i] - max_val);
    }
    sum_exp = max_val + logf(sum_exp);

    for (int i = 0; i < num_classes; i++) {
        output[idx * num_classes + i] = expf(input[idx * num_classes + i] - sum_exp);
    }
}

__global__ void online_normalizer_softmax_kernel(float* input, float* output, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = -INFINITY;
    float prev_max = -INFINITY;
    float denominator = 0;
    for (int i = 0; i < num_classes; i++) {
        prev_max = max_val;
        max_val = max(max_val, input[idx * num_classes + i]);
        denominator = denominator * expf(prev_max - max_val) + expf(input[idx * num_classes + i] - max_val);
    }

    for (int i = 0; i < num_classes; i++) {
        output[idx * num_classes + i] = expf(input[idx * num_classes + i] - max_val) / denominator;
    }
}


__global__ void share_memory_softmax_kernel(float* input, float* output, int num_classes, int num_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE_ROW];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row > num_row) {
        return;
    }
    
    float max_val = -INFINITY;
    float denominator = 0;

    for (int i = tid; i < num_classes; i += blockDim.x) {
        float x = input[row * num_classes + i];
        float new_max = max(max_val, x);
        denominator = denominator * expf(max_val - new_max) + expf(x - new_max);
        max_val = new_max; 
    }
    __syncthreads();

    // Each thread has its own max value 
    sdata[tid] = max_val;
    __syncthreads();

    // Reduce max value
    for (int s = blockDim.x / 2; s > 0; s /= 2) { 
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    } 

    float row_max = sdata[0];
    __syncthreads();

    // Each thread has its own denominator
    sdata[tid] = denominator * expf(max_val - row_max);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) { 
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    float row_denominator = sdata[0];
    __syncthreads();

    for (int i = tid; i < num_classes; i += blockDim.x) {
        output[row * num_classes + i] = expf(input[row * num_classes + i] - row_max) / row_denominator;
    }
}


__global__ void warp_shuffle_softmax_kernel(float* input, float* output, int num_classes, int num_row) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    unsigned mask =  0xffffffff;
    __shared__ float smax[32];
    __shared__ float sdenominator[32];
    if (row > num_row) {
        return;
    }

    float max_val = -INFINITY;
    float denominator = 0;

    for (int i = tid; i < num_classes; i += blockDim.x) {
        float x = input[row * num_classes + i];
        float new_max = max(max_val, x);
        denominator = denominator * expf(max_val - new_max) + expf(x - new_max);
        max_val = new_max; 
    }
    __syncthreads();

    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(mask, max_val, offset);
        float other_denominator = __shfl_down_sync(mask, denominator, offset);
        max_val = max(max_val, other_max);
        denominator = denominator * expf(other_max - max_val) + other_denominator;
    }

    if (lane_id == 0) {
        smax[warp_id] = max_val;
        sdenominator[warp_id] = denominator;
    }
    __syncthreads();

    //Only first warp do the reduction
    if (warp_id == 0) {
        max_val = (tid < blockDim.x / warp_size) ? smax[tid] : -INFINITY;
        denominator = (tid < blockDim.x / warp_size) ? sdenominator[tid] : 0;

        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(mask, max_val, offset);
            float other_denominator = __shfl_down_sync(mask, denominator, offset);
            max_val = max(max_val, other_max);
            denominator = denominator * expf(other_max - max_val) + other_denominator;
        }
        if (tid==0) {
            smax[0] = max_val;
            sdenominator[0] = denominator;
        }
    }

    __syncthreads();

    float row_max = smax[0];
    float row_denominator = sdenominator[0];
    for (int i = tid; i < num_classes; i += blockDim.x) {
        output[row * num_classes + i] = expf(input[row * num_classes + i] - row_max) / row_denominator;
    }
}


torch::Tensor naive_softmax(torch::Tensor input) {
    auto output = torch::empty_like(input, input.options());
    int num_classes = input.size(1);
    int num_samples = input.size(0);
    naive_softmax_kernel<<<(num_samples + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW, BLOCK_SIZE_ROW>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_classes);
    return output;
}


torch::Tensor online_normalizer_softmax(torch::Tensor input) {
    auto output = torch::empty_like(input, input.options());
    int num_classes = input.size(1);
    int num_samples = input.size(0);
    online_normalizer_softmax_kernel<<<(num_samples + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW, BLOCK_SIZE_ROW>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_classes);
    return output;
}

torch::Tensor share_memory_softmax(torch::Tensor input) {
    auto output = torch::empty_like(input, input.options());
    int num_classes = input.size(1);
    int num_samples = input.size(0);
    share_memory_softmax_kernel<<<num_samples, BLOCK_SIZE_ROW>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_classes, num_samples);
    return output;
}

torch::Tensor warp_shuffle_softmax(torch::Tensor input) {
    auto output = torch::empty_like(input, input.options());
    int num_classes = input.size(1);
    int num_samples = input.size(0);
    warp_shuffle_softmax_kernel<<<num_samples, BLOCK_SIZE_COL>>>(input.data_ptr<float>(), output.data_ptr<float>(), num_classes, num_samples);
    return output;
}