/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <cinttypes>
#include <stdio.h>

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
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

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row  < m && col < n) {
        for (int i = 0; i < n; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++) {
                sum += A[row * k + j] * B[j * n + i];
            }
            C[row * n + i] = sum;
        }
    }

}


void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    // const unsigned int BLOCK_SIZE = 16; // Use 16x16 thread blocks
    const unsigned int BLOCK_SIZE = 32; // Use 1x32 thread blocks
    //INSERT CODE HERE
    dim3 dim_grid, dim_block;
    dim_block = dim3(1, BLOCK_SIZE, 1);
    dim_grid = dim3(n, (m + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);

}


