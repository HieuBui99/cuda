// #include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_DIM 1024

__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();

    }
    if (threadIdx.x == 0) {
     *output = input[0];
    }
}

__global__ void SharedMemoryReduction(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t  + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /=2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}


// torch::Tensor simple_sum_reduction(torch::Tensor input, char* method) {
//     auto output = torch::zeros(1, input.options());
//     float* input_ptr = input.data_ptr<float>();
//     float* output_ptr = output.data_ptr<float>();

//     if (strcmp(method, "shared") == 0) {
//         SharedMemoryReduction<<<1, BLOCK_DIM>>>(input_ptr, output_ptr);
//     } else {
//         SimpleSumReductionKernel<<<1, input.size(0) / 2>>>(input_ptr, output_ptr);
//     }
//     return output;

// }


int main() {
    // Size of the input data
    const int size = 1024;
    const int bytes = size * sizeof(float);

    // Allocate memory for input and output on host
    float* h_input = new float[size];
    float* h_output = new float;

    // Initialize input data on host
    for (int i = 0; i < size; i++) {
        h_input[i] = 2.0f; // Example: Initialize all elements to 2
    }

    // Allocate memory for input and output on device
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    SimpleSumReductionKernel<<<1, size / 2>>>(d_input, d_output);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Sum is " << *h_output << std::endl;

    // Cleanup
    delete[] h_input;
    delete h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

