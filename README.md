# GDBMS-Capstone
GPU-accelerated Database Management Systems (GDBMS) research for my YSU Capstone project.

Makefile does not currently work
USE THESE COMMANDS TO COMPILE AND RUN INSTEAD:
`$ nvcc -g -G --gpu-architecture=sm_60 -c kernel.cu -o kernel.o`
`$ nvcc -std=c++11 -L/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcurand -o main kernel.o -Xcompiler -static-libgcc -Xcompiler -static-libstdc++`
`$ ./main`
