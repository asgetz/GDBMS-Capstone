# GDBMS-Capstone
GPU-accelerated Database Management Systems (GDBMS) research for my YSU Capstone project.

Makefile does not work currently. I've been using
`nvcc -g -G -Xcompiler -Wall -lcudart -c kernel.cu -o kernel.o`
and then linking the .o file to a main.exe with nvcc.

<!-- $ nvcc -g -G -Xcompiler -Wall -lcudart -lcurand -c kernel.cu -o kernel.o

$ nvcc --gpu-architecture=sm_70 --device-link kernel.o --output-file link.o -->

<!-- $ nvcc --lib --output-file gpu.lib kernel.o link.o -->
<!-- Above 3 commands use nvcc to compile and link the kernel into its own library. From there, I make use of g++ to merely compile the host code and link to the created library. This was needed because I don't know how to get NVCC to accept hostcode that uses the C++ Boost libraries. Using separate compilers that evaluate to the same executable doesn't work either. So this is the current workaround to this. -->

<!-- $ g++ -g -Wall -std=c++11 -O3 -I. -I"D:\\Boost\\boost_1_69_0"
-I"C:\\PROGRA~1\\NVIDIA~2\\CUDA\\v10.0\\include" -c hostcode.cpp -o hostcode.o

$ g++ -g -Wall -std=c++11 -O3 -L"D:\\git\\GDBMS-Capstone2\\GDBMS-Capstone\\src"
-L"C:\\PROGRA~1\\NVIDIA~2\\CUDA\\v10.0\\lib\\x64"
-L"D:\\Boost\\boost_1_69_0\\stage\\lib"
-lgpu -lcudart -lcudadevrt -lcurand
-lboost_filesystem-mgw73-mt-d-x64-1_69 -lboost_iostreams-mgw73-mt-d-x64-1_69
-lboost_system-mgw73-mt-d-x64-1_69 -o main hostcode.o -->
