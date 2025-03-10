
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda-12.5/include
LD_FLAGS    = -lcudart -L/usr/local/cuda-12.5/lib64
EXE	        = reduce
OBJ	        = main.o
ARCH        = -arch=sm_89

default: $(EXE)

main.o: reduce.cu
	$(NVCC) -c -o $@ reduce.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
