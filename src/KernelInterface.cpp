// Alex Getz

#include <cuda.h>
#include <cuda_runtime.h>
#include <KernelInterface.h>
#include <stdio.h>
#include <stdlib.h>

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

#define MUL(a, b) a*b


/* ///   Host Functions     /// */


/* ///   DEVICE FUNCTIONS   /// */


/* ///   GLOBAL FUNCTIONS   /// */
/**
 * [addKernel description]
 * @method addKernel
 * @param  c         [description]
 * @param  a         [description]
 * @param  b         [description]
 */
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


/**
 * CUDA Kernel designated to carry out the bootstrap of the Base sample.
 * When launched, this kernel will generate a random number generator for
 * each thread that executes with it. These threads, with their random
 * number, will choose a random element from the original sample. Then the
 * kernel will construct an array of x_bar values by summation amongst the
 * threads.
 * @method bootstrap
 * @param  out       This is the output array, it will contain the x_bar values
 * @param  d_sample  This is the original sample that gets copied onto the GPU
 * @param  state     An array of pointers to unique cuRAND instances for threads
 * @return           return.
 */
__global__ void bootstrap(unsigned int *out, int *d_sample)
{
    /*  */
    // unsigned int tid = threadIdx.x;
    // unsigned int block = blockIdx.x;
    // unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    // unsigned int ts;
    //
    // curandState localState = state[idx];
    // // curand_init(1234,0,idx,&localState);
    // // __syncthreads();
    //
    //
    // ts = curand(&localState);
    // out[idx]=d_sample[ts%100];
    // __syncthreads();

    /*
     * Attempt to separate the blocks of 128 threads into
     * logical groupings of 100 for summation
     */
    // int x = idx%100;
    // int tx = idx-x;
    // int y = idx%128;
    // int ty = idx-y;
    // if(ty>tx){
    //     out[idx]=99999;
    // }else{
    //     out[idx]=(idx%100)+1;
    // }
}


/**
 * [parkmillerKernel description]
 * @method parkmillerKernel
 * @param  d_Output         [description]
 * @param  seed             [description]
 * @param  cycles           [description]
 * @param  N                [description]
 */
static __global__ void parkmillerKernel(int *d_array, unsigned int seed, \
                                        int cycles, unsigned int N)
{
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);
    double const a    = 16807;      //ie 7**5
    double const m    = 2147483647; //ie 2**31-1
    double const reciprocal_m = 1.0/m;

    for(unsigned int pos = tid; pos < N; pos += threadN){
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for(int i = 0; i < cycles; i++) {
            double temp = data * a;
            result = (int) (temp - m * floor ( temp * reciprocal_m ));
            data = result;
        } //endfor

        //d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
        //d_Output[MUL(threadIdx.y, N) + pos] = result;
        d_array[pos] = result;
    }
}

KernelInterface::KernelInterface(float r, int sSize, int pSize, int bsnum)
: ratio(r), Ssize(sSize), PopSize(pSize), bStraps(bsnum)
{
    /**/
}

void KernelInterface::parkmillercall(unsigned int threads, unsigned int blocks){
    checkCuda( cudaMalloc((void**)&randArray,PopSize*sizeof(int)));
    parkmillerKernel<<<blocks,threads>>>(randArray,1234,bStraps,PopSize);
    cudaFree(randArray);
}

// __device__ void addKernelGPU(unsigned int blocknum, \
//                                       unsigned int threadnum, \
//                                       int *c, \
//                                       const int *a, \
//                                       const int *b)
// {
//     /* */
// }
//
// __device__ void bootstrapGPU(unsigned int blocknum, \
//                                       unsigned int threadnum, \
//                                       unsigned int *output, \
//                                       int *sampleInput)
// {
//     /* Call bootstrap in here.
//      * Kernel invocation as well as perhaps Allocation, cudacopies
//      * memsets, and other such things.
//      */
//     //
//     // //////// Copy from Host to device
//     // cudaMemcpy(d_Base,BaseSample,bytes,cudaMemcpyHostToDevice);
//     // // cudaMemcpy(d_mean,h_mean,bytes,cudaMemcpyHostToDevice);
//     //
//     // //////// Copy back from device to host
//     // cudaMemcpy(h_mean,d_mean,313*128*sizeof(int),cudaMemcpyDeviceToHost);
//     // for(int c=0;c<313*128;++c){
//     //     std::cout<<"element "<<c<<" : "<<h_mean[c]<<std::endl;
//     // }
// }

// __device__ void parkmillerGPU(unsigned int seed, \
//                                        int cycles, \
//                                        unsigned int grid_size, \
//                                        unsigned int block_size, \
//                                        unsigned int N)
// {
//     /* Inside description here */
//     // checkCuda( cudaMallocHost((void**)&BaseSample,MAX_ENTRIES*sizeof(int)));
// }
