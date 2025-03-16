nvcc -o benchmark file.cu
ncu benchmark


Performance checklist
* Control divergence (if-else in thread)
* Memory divergence 
* Minimize global memory access
* THread coarsening (Try to get threads to do more work)