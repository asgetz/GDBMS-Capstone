UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL -Xlinker -framework,GLUT
else
	LDFLAGS += -L/c/ProgramData/NVIDIA~1/CUDASA~1/v10.0/common/lib/x64/freeglut.lib
	LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif

NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
NVCC_FLAGS = -g -G -Xcompiler -Wall -Wno-deprecated-declarations
INC = -I/c/ProgramData/NVIDIA~1/CUDASA~1/v10.0/common/inc

all: main.exe

main.exe: main.o kernel.o
	$(NVCC) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernel.h interactions.h
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f *.o *.exe
