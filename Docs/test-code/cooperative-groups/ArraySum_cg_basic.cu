// Created by Alex Getz

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <assert.h>
#include <iostream>

namespace cg = cooperative_groups;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if(result!=cudaSuccess){
        fprintf(stderr,"CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result==cudaSuccess);
    }
#endif
    return result;
}

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

///// Host Functions

///// Device Functions
__device__ int ReduceSum(cg::thread_group tg, int *temp, int value){
    int lane = tg.thread_rank();

    for(int i=tg.size()/2;i>0;i/=2){
        temp[lane]=value;
        tg.sync();
        if(lane<i){ value += temp[lane+i]; }
        tg.sync();
    }
    return value;
}

__device__ int ThreadSum(int *input, int n){
    int sum=0;
    for(int i=threadIdx.x+(blockIdx.x*blockDim.x);i<n/4;i+=blockDim.x*gridDim.x)
    {
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

///// Device Kernels
__global__ void SumKernel(int *input, int *output, int n){
    // unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    int my_sum = ThreadSum(input, n);

    extern __shared__ int temp[];
    auto g = cg::this_thread_block();
    int block_sum = ReduceSum(g, temp, my_sum);

    if (g.thread_rank() == 0) atomicAdd(output, block_sum);


    // output[idx]=input[idx];
}

/////

int main(int argc, char const *argv[]) {
    // Should expand to do some command line parsing of arrays in future

    int n = 1<<24;
    int blockSize = 256;
    int nBlocks = (n + blockSize - 1) / blockSize;
    int sharedBytes = blockSize * sizeof(int);

    int *sum, *data;
    cudaMallocManaged(&sum, sizeof(int));
    cudaMallocManaged(&data, n * sizeof(int));
    std::fill_n(data, n, 1); // initialize data
    cudaMemset(sum, 0, sizeof(int));

    SumKernel<<<nBlocks, blockSize, sharedBytes>>>(data, sum, n);

    // const unsigned int arrSize = 300;
    // const unsigned int bytes = 300*sizeof(int);
    // int *h_input, *d_input, *h_output, *d_output;
    //
    // checkCuda( cudaMallocHost((int**)&h_input,bytes));
    // cudaMalloc((void**)&d_input,bytes);
    // checkCuda( cudaMallocHost((void**)&h_output,bytes));
    // checkCuda( cudaMalloc((void**)&d_output,bytes));
    // printf("Before\n");
    // for(int i=0;i<300;++i){
    //     h_input[i]=i;
    //     std::cout << h_input[i] << '\n';
    // }
    //
    // //////////
    // checkCuda( cudaMemcpy(d_input,h_input,bytes,cudaMemcpyHostToDevice));
    // //////////
    // SumKernel<<<3,128>>>(d_input,d_output,arrSize);
    // //////////
    // checkCuda( cudaMemcpy(h_output,d_output,bytes,cudaMemcpyDeviceToHost));
    // //////////
    //
    // printf("After\n");
    // for(int a=0;a<300;++a){
    //     std::cout<<h_output[a]<<std::endl;
    // }
    //
    // cudaFree(d_input);
    // cudaFree(d_output);
    // cudaFreeHost(h_input);
    // cudaFreeHost(h_output);
    return 0;
}
