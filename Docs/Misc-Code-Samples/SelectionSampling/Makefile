# nvcc.exe -ccbin "D:\VisualStudio...\14.16.27023\bin\HostX86\x86" -x cu
# -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" -G
# --keep-dir Debug -maxrregcount=0
# --machine 32 --compile
# -DWIN32 -D_DEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Od  /FS /Zi /RTC1 /MDd " -o Debug\SelectionSampling.cu.obj
SRC = SelectionSamplingExample.cu CustomSelectionSampling.cu

# OBJ = $(SRC:.cu=.o)

# Include Directories
# INCLUDES = -I "D:\Boost\boost_1_69_0"

# Compiler flags
NVCC_FLAGS = -g -G -Xcompiler -Wall -lcurand
CFLAGS = -g -Wall
CXXFLAGS = -g -std=c++11 -O3 -Wall

# Compiler
NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
CC = gcc
CXX = g++

# Library Paths
# LIBS = -L/d/Boost/boost_1_69_0/stage/lib

# Compile flags
# LDFLAGS = -L/d/Boost/boost_1_69_0/stage/lib -l boost_filesystem-mgw73-mt-d-x64-1_69 -l boost_iostreams-mgw73-mt-d-x64-1_69 -l boost_system-mgw73-mt-d-x64-1_69

GSS: $(SRC)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# %.o: $(SRC)
# 	$(NVCC) $(NVCC_FLAGS) $< -c $@

clean:
	rm -f $(OBJ)
