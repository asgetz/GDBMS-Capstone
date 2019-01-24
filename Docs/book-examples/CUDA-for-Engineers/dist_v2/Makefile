NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: main.exe

main.exe: main.o aux_functions.o
	$(NVCC) $^ -o $@

main.o: main.cpp aux_functions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

aux_functions.o: aux_functions.cpp aux_functions.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
