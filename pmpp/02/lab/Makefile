
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda-12.5/include
LD_FLAGS    = -lcudart -L/usr/local/cuda-12.5/lib64
EXE	        = vecadd
OBJ	        = main.o support.o
ARCH        = -arch=sm_89

default: $(EXE)

main.o: main.cu kernel.cu support.h
	$(NVCC) $(ARCH) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) $(ARCH) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(ARCH) $(OBJ) -o $(EXE) $(LD_FLAGS)

test:
	$(NVCC) $(ARCH) -c -o test vec_add_v1.cu $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
