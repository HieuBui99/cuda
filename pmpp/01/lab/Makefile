
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda-12.5/include
LD_FLAGS    = -lcudart -L/usr/local/cuda-12.5/lib64
EXE	        = device-query
OBJ	        = main.o

default: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
