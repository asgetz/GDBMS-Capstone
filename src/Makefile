# NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
# NVCC_FLAGS = -g -G -Xcompiler -Wall -lcurand
#
# main.exe: kernel.cu
# 	$(NVCC) $(NVCC_FLAGS) $^ -o $@
#
# clean:
# 	rm -f *.o *.exe


SRC = hostcode.cpp kernel.cu

# OBJ = $(SRC:.cpp=.o)
# OBJ = hostcode.o kernel.o

# Include Directories
INCLUDES = 	-I"D:\Boost\boost_1_69_0" \
			-I"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\include" \
			-I"C:\ProgramData\NVIDIA~1\CUDASA~1\v10.0\common\inc\GL"

# Compiler flags
CFLAGS = -g -Wall
CXXFLAGS = -g -std=c++11 -O3 -Wall
NVCC_FLAGS = -g -G -Xcompiler -Wall -lcudart -lcurand

# Compiler
NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
CC = gcc
CXX = g++
CL = /c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.15.26726/bin/Hostx64/x64/cl.exe


# Library Paths
LIBS =	-L"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\lib\x64" \
		-L"D:\Boost\boost_1_69_0\stage\lib"
		# -L"C:\ProgramData\NVIDIA~1\CUDASA~1\v10.0\common\lib\x64"
		# C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\lib\x64

# Compile flags
LDFLAGS =	-lboost_filesystem-mgw73-mt-d-x64-1_69 \
			-lboost_iostreams-mgw73-mt-d-x64-1_69  \
			-lboost_system-mgw73-mt-d-x64-1_69
# LDFLAGS += -lfreeglut -lglew64

## With boost
all: main.exe

main.exe: hostcode.o kernel.o
	$(CL) $(LIBS) $(LDFLAGS) $^ -o $@

hostcode.o: hostcode.cpp kernel.h
	$(CL) $(CFLAGS) $(INCLUDES) -c $< -o $@

kernel.o: kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# kernel.cu.o: kernel.cu
# 	$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

clean:
	rm -f *.o *.exe
