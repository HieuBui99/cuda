## Data parallelism

Computation work to be
performed on different parts of the dataset can be done independently of eachother and thus can be done in parallel with each other.

## CUDA C Program Structure

The structure of a CUDA C program reflects the coexistence of a host (CPU)
and one or more devices (GPUs). Each CUDA C source file can
have a mixture of host code and device code.

The device code includes functions, or **kernels**

Program execution:
1.  Start with host code
2. Kernel function is called -> launch threads. Threads launched by kernel are called grid



`Divergence`: Some threads does not need to do work cause we initialize block size in power of 2


`__global__`: special keyword that tells ther compiler this is a kernel function 


Use number of threads as  multiple of 32 because there's a penalty (not huge)

Use `ceil` cause C will truncate float


```bash
nvcc  -arch=sm_89 -c -o main.o main.cu -O3 -I/usr/local/cuda-12.5/include
nvcc  -arch=sm_89 -c -o support.o support.cu -O3 -I/usr/local/cuda-12.5/include
nvcc  -arch=sm_89main.o support.o -o vecadd -lcudart -L/usr/local/cuda-12.5/lib64
```

Need to add `-arch=sm_89` to specify cuda capability. Otherwise the kernel wont't launch