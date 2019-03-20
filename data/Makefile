NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
NVCC_FLAGS = -g -G -Xcompiler -Wall
datagen.exe: datagen.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@
