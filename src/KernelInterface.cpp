// Alex Getz

#include <cuda.h>
#include <cuda_runtime.h>
#include <KernelInterface.h>
#include <stdio.h>
#include <stdlib.h>

KernelInterface::KernelInterface()
{
    /* Build whatever.
     * Probably should handle all allocation in here. Perhaps.
     * Maybe park miller in here
     */
    //

    y = 20;
    asize = y*sizeof(int);
    for (int i=0; i<y; i++){ n[i] = i; }

    // ///// Allocate space for prng states on device
    // CUDA_CALL(cudaMalloc((void**)&devStates, 313*128*sizeof(curandState)));

    ////// Allocate memory in GPU for the sample data
    // checkCuda( cudaMallocHost((void**)&pData_h,MAX_ENTRIES*sizeof(int)));
    // cudaMalloc((void**)&pData_d,MAX_ENTRIES*sizeof(int));
    // checkCuda( cudaMallocHost((int**)&BaseSample,bytes));
    // cudaMalloc((void**)&d_Base,bytes);
    // checkCuda( cudaMallocHost((void**)&h_mean,313*128*sizeof(int)));
    // checkCuda( cudaMalloc((void**)&d_mean,313*128*sizeof(int)));
}

__device__ void addKernelGPU(unsigned int blocknum, \
                                      unsigned int threadnum, \
                                      int *c, \
                                      const int *a, \
                                      const int *b)
{
    /* */
}

__device__ void bootstrapGPU(unsigned int blocknum, \
                                      unsigned int threadnum, \
                                      unsigned int *output, \
                                      int *sampleInput)
{
    /* Call bootstrap in here.
     * Kernel invocation as well as perhaps Allocation, cudacopies
     * memsets, and other such things.
     */
    //
    // //////// Copy from Host to device
    // cudaMemcpy(d_Base,BaseSample,bytes,cudaMemcpyHostToDevice);
    // // cudaMemcpy(d_mean,h_mean,bytes,cudaMemcpyHostToDevice);
    //
    // //////// Copy back from device to host
    // cudaMemcpy(h_mean,d_mean,313*128*sizeof(int),cudaMemcpyDeviceToHost);
    // for(int c=0;c<313*128;++c){
    //     std::cout<<"element "<<c<<" : "<<h_mean[c]<<std::endl;
    // }
}

__device__ void parkmillerGPU(int *d_Output, \
                                       unsigned int seed, \
                                       int cycles, \
                                       unsigned int grid_size, \
                                       unsigned int block_size, \
                                       unsigned int N)
{
    /* Inside description here */
}
