#################################################################

compile:
	nvcc -Xptxas -v -g -G --gpu-architecture=sm_35 -c kernel.cu -o kernel.o
	nvcc -v -std=c++11 -L/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcurand -o main kernel.o -Xcompiler -static-libgcc -Xcompiler -static-libstdc++
#################################################################




# ###############################################################
# #####################                   Most Up-To-Date Attempt
# ##################### Geared towards the SARAH3 CentOS-7 System
# kernel.o:
#	@nvcc -g -G --gpu-architecture=sm_35 -c kernel.cu -o kernel.o &&
#	@nvcc -std=c++11 -L/usr/local/cuda/include -L/usr/local/cuda/lib64 \
#		-lcudart -lcurand -o main kernel.o -Xcompiler -static-libgcc \
#		-Xcompiler -static-libstdc++
#
#	echo "First time compilation successful"
#
# recompile: kernel.o
#	@nvcc -g -G --gpu-architecture=sm_35 -c kernel.cu -o kernel.o &&
#	@nvcc -std=c++11 -L/usr/local/cuda/include -L/usr/local/cuda/lib64 \
#		-lcudart -lcurand -o main kernel.o -Xcompiler -static-libgcc \
#		-Xcompiler -static-libstdc++
#
#	echo "Recompilation successful"
# ###############################################################





# ########################### Phase 2 Attempt 3
# NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
#
# SRC = hostcode.cpp KernelInterface.cpp
#
# # All device code
# OBJS_1 = kernel.o KernelInterface.o
#
# # All non-device code
# OBJS_2 = hostcode.o KernelInterface.o
#
# DeviceCODE: $(OBJS_1)
# 	$(NVCC) -arch=sm_60 $(OBJS_1) -o DeviceCODE
#
# $(OBJS_1): $(SRC)
#	$(NVCC) -x cu -arch=sm_60 -I. -dc $< -o $@
#
# clean:
#	rm -f *.o DeviceCODE
######################################################



# $ nvcc -x cu --gpu-architecture=sm_60 -I. --device-c KernelInterface.cpp
# $ nvcc --gpu-architecture=sm_60 --device-link KernelInterface.obj --output-file link1.o
# $ g++ -g -std=c++11 -I. -I"D:\Boost\boost_1_69_0" -I"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\include" -c hostcode.cpp -o hostcode.o
# $ g++ -g -std=c++11 -I. -I"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\include" -c KernelInterface.cpp -o KernelInterface.o
# $ nvcc -g -G -I. -c kernel.cu -o kernel.o
# $ nvcc --lib --output-file libgpu.a kernel.o link1.o
# $ g++ -g -o main.exe hostcode.o KernelInterface.o -L"D:\git\GDBMS-Capstone2\GDBMS-Capstone\src" -L"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\lib\x64" -L"D:\Boost\boost_1_69_0\stage\lib" -lgpu -lcudart -lboost_filesystem-mgw73-mt-d-x64-1_69 -lboost_iostreams-mgw73-mt-d-x64-1_69 -lboost_system-mgw73-mt-d-x64-1_69



# ######################### Phase 2 Attempt 2
# CXX = g++
# CC = gcc
# LINK = g++ -fPIC
# NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
#
# # Important Install Paths
# CUDA_INSTALL_PATH = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0
# # BOOST_PATH = /d/Boost/boost_1_69_0
#
# # Includes
# INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include
# # BOOST_INCLUDE = -I$(BOOST_PATH)
#
# # Common Flags
# COMMON_FLAGS += $(INCLUDES)
# NVCC_FLAGS += $(COMMON_FLAGS)
# CXX_FLAGS += $(COMMON_FLAGS) #$(BOOST_INCLUDE)
# C_FLAGS += $(COMMON_FLAGS) #$(BOOST_INCLUDE)
#
# # Library Paths
# CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib/x64 -lcuda -lcudart -lcudadevrt
# # BOOST_LIBS:= -L$(BOOST_PATH)/stage/lib -lboost_filesystem-mgw73-mt-d-x64-1_69 -lboost_iostreams-mgw73-mt-d-x64-1_69 -lboost_system-mgw73-mt-d-x64-1_69
#
# #################
# OBJS = kernel.cu.o hostcode.cpp.o
# TARGET = exec
# LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(CUDA_LIBS) #$(BOOST_LIBS)
#
# .SUFFIXES: .c .cpp .cu .o
#
# %.c.o: %.c
# 	$(CC) $(C_FLAGS) -c $< -o $@
#
# %.cu.o: %.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
#
# %.cpp.o: %.cpp
# 	$(CXX) $(CXX_FLAGS) -c $< -o $@
#
# $(TARGET): $(OBJS) Makefile
# 	$(LINKLINE)
#
# clean:
# 	rm -f *.o *.exe



######################### Phase 2 Attempt 1
# # CC = gcc
# # CCPP = g++
# # NVCC = /vol/cuda/8.0.61/bin/nvcc
# CC = gcc
# CXX = g++
# NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
#
# #preprocessor defines
# DEFINES = -DUSE_ZLIB
#
# OPTFLAGS = -Ofast
#
# #CFLAGS = -ansi -pedantic -Wall -Werror -fPIC
# #CFLAGS = -ansi -pedantic -Wall -fPIC
# CFLAGS = -ansi $(OPTFLAGS)
# NVFLAGS = -x c++ -Wno-deprecated-gpu-targets -ccbin g++ -Xcompiler -ansi -Xcompiler -Ofast
#
# NVLINKFLAGS = -L/vol/cuda/8.0.61/lib64/ -lcuda -lcudart
# #NVLINKFLAGS += -L/vol/cuda/8.0.61/lib -lcuda -lcudart
#
# CPPFLAGS = $(OPTFLAGS) -I./include $(DEFINES)
# LINKFLAGS = -lz
#
# all : simple
#
# objs =  src/tourtre.o     \
#     src/ctArc.o       \
#     src/ctBranch.o    \
#     src/ctComponent.o \
#     src/ctNode.o      \
#     src/ctQueue.o     \
#     src/ctNodeMap.o   \
#     examples/simple/obj/Data.o \
#     examples/simple/obj/Mesh.o \
#     examples/simple/obj/main.o
#
# simple : $(objs)
#     $(CCPP)  $(CPPFLAGS) -o simple $^ $(LINKFLAGS) $(NVLINKFLAGS)
#
# src/tourtre.o : src/tourtre.c include/tourtre.h src/ctMisc.h include/ctArc.h include/ctNode.h src/ctComponent.h include/ctNode.h src/ctQueue.h src/ctAlloc.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# src/ctArc.o : src/ctArc.c include/tourtre.h src/ctMisc.h include/ctArc.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# src/ctBranch.o : src/ctBranch.c include/tourtre.h src/ctMisc.h include/ctBranch.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# src/ctComponent.o : src/ctComponent.c include/tourtre.h src/ctMisc.h src/ctComponent.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# src/ctNode.o : src/ctNode.c include/tourtre.h src/ctMisc.h include/ctNode.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# src/ctQueue.o : src/ctQueue.c include/tourtre.h src/ctMisc.h src/ctQueue.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# src/ctNodeMap.o : src/ctNodeMap.c src/ctNodeMap.h include/ctNode.h src/ctQueue.h src/sglib.h
#     $(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# examples/simple/obj/Data.o: examples/simple/src/Data.cpp examples/simple/src/Data.h examples/simple/src/Global.h
#     $(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
# examples/simple/obj/Mesh.o: examples/simple/src/Mesh.cpp examples/simple/src/Data.h examples/simple/src/Global.h examples/simple/src/Mesh.h
#     $(NVCC) $(NVFLAGS) -c $< -o $@
#
# examples/simple/obj/main.o: examples/simple/src/main.cpp examples/simple/src/Data.h examples/simple/src/Global.h examples/simple/src/Mesh.h
#     $(CCPP) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
#
#
# clean :
#     -rm -rf src/*.o examples/simple/obj/*.o doc/html



############## Phase 1 makefile
# SRC = hostcode.cpp kernel.cu
#
# # OBJ = $(SRC:.cpp=.o)
# # OBJ = hostcode.o kernel.o
#
# # Include Directories
# INCLUDES = 	-I"D:\Boost\boost_1_69_0" \
# 			-I"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\include" \
# 			-I"C:\ProgramData\NVIDIA~1\CUDASA~1\v10.0\common\inc\GL"
#
# # Compiler flags
# CFLAGS = -g -Wall
# CXXFLAGS = -g -std=c++11 -O3 -Wall
# NVCC_FLAGS = -g -G -Xcompiler -Wall -lcudart -lcurand
#
# # Compiler
# NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
# CC = gcc
# CXX = g++
# CL = /c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.15.26726/bin/Hostx64/x64/cl.exe
#
#
# # Library Paths
# LIBS =	-L"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\lib\x64" \
# 		-L"D:\Boost\boost_1_69_0\stage\lib"
# 		# -L"C:\ProgramData\NVIDIA~1\CUDASA~1\v10.0\common\lib\x64"
# 		# C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\lib\x64
# nvcc --gpu-architecture=sm_50 --device-c a.cu b.cu
# nvcc --gpu-architecture=sm_50 --device-link a.o b.o --output-file link.o
# nvcc --lib --output-file libgpu.a a.o b.o link.o
# g++ host.o --library=gpu --library-path=<path> \
#     --library=cudadevrt --library=cudart
# -L"D:\git\GDBMS-Capstone2\GDBMS-Capstone\src"
#
# # Compile flags
# LDFLAGS =	-lboost_filesystem-mgw73-mt-d-x64-1_69 \
# 			-lboost_iostreams-mgw73-mt-d-x64-1_69  \
# 			-lboost_system-mgw73-mt-d-x64-1_69
# # LDFLAGS += -lfreeglut -lglew64
#
# ## With boost
# main.exe: hostcode.o
# 	g++ -g -Wall -std=c++11 -O3
# 	-L"D:\git\GDBMS-Capstone2\GDBMS-Capstone\src"
# 	-L"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\lib\x64"
# 	-L"D:\Boost\boost_1_69_0\stage\lib"
# 	-lgpu -lcudart -lcudadevrt -lcurand
# 	-lboost_filesystem-mgw73-mt-d-x64-1_69
# 	-lboost_iostreams-mgw73-mt-d-x64-1_69
# 	-lboost_system-mgw73-mt-d-x64-1_69
# 	-o main hostcode.o
#
# hostcode.o: hostcode.cpp
# 	g++ -g -Wall -std=c++11 -O3 -I. -I"D:\Boost\boost_1_69_0"
# 	-I"C:\PROGRA~1\NVIDIA~2\CUDA\v10.0\include" -c hostcode.cpp -o hostcode.o
#
# gpu.lib: kernel.o link.o
# 	$(NVCC) --lib --output-file gpu.lib kernel.o link.o
#
# link.o: kernel.o
# 	$(NVCC) --gpu-architecture=sm_70 --device-link kernel.o --output-file link.o
#
# kernel.o: kernel.cu kernel.h
# 	$(NVCC) -g -G -Xcompiler -Wall -lcudart -lcurand -c kernel.cu -o kernel.o
#
# clean:
# 	rm -f *.o *.exe


############### Original Make
# NVCC = /c/PROGRA~1/NVIDIA~2/CUDA/v10.0/bin/nvcc.exe
# NVCC_FLAGS = -g -G -Xcompiler -Wall -lcurand
#
# main.exe: kernel.cu
# 	$(NVCC) $(NVCC_FLAGS) $^ -o $@
#
# clean:
# 	rm -f *.o *.exe
