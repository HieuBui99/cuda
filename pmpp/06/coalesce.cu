#include <cuda_runtime.h>
#include <iostream>

__global__ void copyDataNonCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[(index * 2) % n];
    }
}

__global__ void copyDataCoalesced(float *in, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = in[index];
    }
}

void initializeArray(float *arr, int n) {
    for(int i = 0; i < n; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

int main() {
    const int n = 1 << 24; // Increase n to have a larger workload
    float *in, *out;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocManaged(&in, n * sizeof(float));
    cudaMallocManaged(&out, n * sizeof(float));

    initializeArray(in, n);
    
    // Ensure data is on device before timing
    cudaDeviceSynchronize();

    // int blockSize = 128; // Define block size
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Time non-coalesced kernel
    cudaEventRecord(start);
    copyDataNonCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Non-coalesced kernel time: " << milliseconds << " ms" << std::endl;

    // Reset output array
    initializeArray(out, n);
    cudaDeviceSynchronize();

    // Time coalesced kernel
    cudaEventRecord(start);
    copyDataCoalesced<<<numBlocks, blockSize>>>(in, out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Coalesced kernel time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(in);
    cudaFree(out);

    return 0;
}