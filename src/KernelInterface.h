#ifndef KERNELINTERFACE_H
#define KERNELINTERFACE_H

#include <stdio.h>
#include <stdlib.h>

// /**/
// void addKernelCall(unsigned int bnum, unsigned int tnum, int *c, \
//                    const int *a, const int *b);
//
// /**/
// void bootstrapCall(unsigned int bnum, unsigned int tnum, \
//                   unsigned int *output, int *sampleInput);
//
// /**/
// void parkmillerCall(int *Out, unsigned int seed, int cycles, \
//                    unsigned int grid_size, unsigned int block_size, \
//                    unsigned int N);


class KernelInterface
{
public:
    int n[20];
    int y;
    int asize;

    KernelInterface();

    __device__ void addKernelGPU(unsigned int blocknum, \
                                          unsigned int threadnum, \
                                          int *c, const int *a, \
                                          const int *b);

    __device__ void bootstrapGPU(unsigned int blocknum, \
                                          unsigned int threadnum, \
                                          unsigned int *output, \
                                          int *sampleInput);

    __device__ void parkmillerGPU(int *d_Output, \
                                           unsigned int seed, \
                                           int cycles, \
                                           unsigned int grid_size, \
                                           unsigned int block_size, \
                                           unsigned int N);
};


#endif



// #ifdef __MINGW32__
//
// #if 0
// extern "C" {
// int __security_cookie;
// }
//
// extern "C" void _fastcall __security_check_cookie(int i) {
// //do nothing
// }
// #endif
// extern "C" void _chkstk() {
// //do nothing
// }
// extern "C" void _allmul() {
// //do nothing
// }
// #endif //__MINGW32__

// #include <device_launch_parameters.h>

// #ifdef __CUDACC__
// #define CUDA_CALLABLE_MEMBER __device__ __host__
// #else
// #define CUDA_CALLABLE_MEMBER
// #endif




// /**
//  * [KernelInterface::bootstrapGPU description]
//  * @method KernelInterface::bootstrapGPU
//  * @param  blocknum                      [description]
//  * @param  threadnum                     [description]
//  * @param  output                        [description]
//  * @param  sampleInput                   [description]
//  */
// // __device__ __host__
// void KernelInterface::bootstrapGPU(unsigned int blocknum, \
//                                    unsigned int threadnum, \
//                                    unsigned int *output, \
//                                    int *sampleInput)
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
//     //
//     // //////// Copy back from device to host
//     // cudaMemcpy(h_mean,d_mean,313*128*sizeof(int),cudaMemcpyDeviceToHost);
//     // for(int c=0;c<313*128;++c){
//     //     std::cout<<"element "<<c<<" : "<<h_mean[c]<<std::endl;
//     // }
// }

// #ifdef __MINGW32__
//
// #if 0
// extern "C" {
// int __security_cookie;
// }
//
// extern "C" void _fastcall __security_check_cookie(int i) {
// //do nothing
// }
// #endif
// extern "C" void _chkstk();
// extern "C" void _allmul();
// #endif //__MINGW32__



// /**
//  * [KernelInterface::KernelInterface description]
//  * @method KernelInterface::KernelInterface
//  */
// // __device__ __host__
// KernelInterface::KernelInterface()
// {
    // /* Build whatever.
    //  * Probably should handle all allocation in here. Perhaps.
    //  * Maybe park miller in here
    //  */
    // //
    //
    // y = 20;
    // asize = y*sizeof(int);
    // for (int i=0; i<y; i++){ n[i] = i; }
    //
    // // ///// Allocate space for prng states on device
    // // CUDA_CALL(cudaMalloc((void**)&devStates, 313*128*sizeof(curandState)));
    //
    // ////// Allocate memory in GPU for the sample data
    // // checkCuda( cudaMallocHost((void**)&pData_h,MAX_ENTRIES*sizeof(int)));
    // // cudaMalloc((void**)&pData_d,MAX_ENTRIES*sizeof(int));
    // // checkCuda( cudaMallocHost((int**)&BaseSample,bytes));
    // // cudaMalloc((void**)&d_Base,bytes);
    // // checkCuda( cudaMallocHost((void**)&h_mean,313*128*sizeof(int)));
    // // checkCuda( cudaMalloc((void**)&d_mean,313*128*sizeof(int)));
// }



//////////////////////////////////////////////////////////////////////////////
/* This function comes from the NVidia Developer Blog entitled:
 * "How to Optimize Data Transfers in CUDA C/C++"
 * link: https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 */
//////////////////////////////////////////////////////////////////////////////
// void profileCopies(int *h_sample, int *h_b, int *d,
//                    unsigned int num,
//                    char *desc)
// {
//     printf("\n%s transfers\n", desc);
//
//   unsigned int bytes = num * sizeof(int);
//
//   // events for timing
//   cudaEvent_t startEvent, stopEvent;
//
//   checkCuda( cudaEventCreate(&startEvent) );
//   checkCuda( cudaEventCreate(&stopEvent) );
//
//   checkCuda( cudaEventRecord(startEvent, 0) );
//   checkCuda( cudaMemcpy(d, h_sample, bytes, cudaMemcpyHostToDevice) );
//   checkCuda( cudaEventRecord(stopEvent, 0) );
//   checkCuda( cudaEventSynchronize(stopEvent) );
//
//   float time;
//   checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
//   printf("  Host to Device bandwidth (bytes/s): %f\n", bytes / time);
//
//   checkCuda( cudaEventRecord(startEvent, 0) );
//   checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
//   checkCuda( cudaEventRecord(stopEvent, 0) );
//   checkCuda( cudaEventSynchronize(stopEvent) );
//
//   checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
//   printf("  Device to Host bandwidth (bytes/s): %f\n", bytes / time);
//
//   for (int i = 0; i < num; ++i) {
//     if (h_sample[i] != h_b[i]) {
//       printf("*** %s transfers failed ***\n", desc);
//       break;
//     }
//   }
//
//   // clean up events
//   checkCuda( cudaEventDestroy(startEvent) );
//   checkCuda( cudaEventDestroy(stopEvent) );
// }
//////////////////////////////////////////////////////////////////////////////




// __device__
// int variance(){/* Calculates variance */}
//
// __device__
// float error(){/* Calculate Margin of error */}
//
// __global__
// void bootstrapKernel(int *V_out, float *E_out, int *S_in){
// 	/**/
// }
//
// void Boostrap(int *SampleBase, int *out, float *error){
//
// 	int *S[]={0};
// 	cudaMalloc(&SampleBase,sizeof(int));
// 	cudaMalloc(&out,sizeof(int));
// 	cudaMalloc(&error,sizeof(float));
//
// 	cudaMemcpy(S,SampleBase,sizeof(int),cudaMemcpyHostToDevice);
//
// 	distanceKernel<<< /*something*/ , /*something*/ >>>(S);
//
// 	cudaMemcpy(V_out,out,sizeof(),cudaMemcpyDeviceToHost);
// 	cudaMemcpy(E_out,error,sizeof(float),cudaMemcpyDeviceToHost);
//
// 	return;
// }




//int *in = (int*)calloc(n,sizeof(int));
//int *out = (int*)calloc(n,sizeof(int));


//    unsigned int bytes = BaseSample.capacity()*sizeof(int) + sizeof(BaseSample);




//    std::cout<<"\n\n\n\nTotal Byte allocation for the Base Sample: ";
//    std::cout<<BaseSample.capacity()*sizeof(int) + sizeof(BaseSample)<<std::endl;
//    std::cout<<"Typical allocation amount: "<<n*sizeof(int)<<"\n\n\n\n";

    ///////////////////////////////////////////////
//    int SampleDataColumn[n];
//    int SampleDataIndex[n];
//    int rand_index[n];
//    std::string entries[N];
//    int numEntries[N];
//    std::ifstream fs("../data/data1.txt");
//    if(!fs){std::cerr<<"Cannot open the data file!"<<std::endl;}
//    else{
//        for(auto i=0;i<std::size(entries);++i){
//            std::getline(fs,entries[i]);
//            numEntries[i]=std::atoi(entries[i].c_str());
//        }
//    }
//    fs.close();

    /*{742,586,999,100,112,829,417,283,333,444,964,30};*/

//    bool exists = std::any_of(  sKey.begin(),
//                                sKey.end(),
//                                [&](int x){return x==temp;}
//                                );



//thrust::host_vector<int> BaseSample(n);




// __global__ void reduce1(int *g_idata, int *g_odata){
//     extern __shared__ int sdata[];
//
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     sdata[tid]=g_idata[i];
//     __syncthreads();
//
//     for(unsigned int s=1;s<blockDim.x;s*=2){
//         if(tid % (2*s) == 0){
//             sdata[tid] += sdata[tid+s];
//         }
//         __syncthreads();
//     }
//
//     if(tid == 0){ g_odata[blockIdx.x] = sdata[0];}
// }
//
// __global__ void reduce2(int *g_idata, int *g_odata){
//     extern __shared__ int sdata[];
//
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     sdata[tid]=g_idata[i];
//     __syncthreads();
//
//     for(unsigned int s=1;s<blockDim.x;s*=2){
//         int index=2*s*tid;
//         if(index<blockDim.x){
//             sdata[index] += sdata[index+s];
//         }
//         __syncthreads();
//     }
//
//     if(tid == 0){ g_odata[blockIdx.x] = sdata[0];}
// }
//
// __global__ void reduce3(int *g_idata, int *g_odata){
//     extern __shared__ int sdata[];
//
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//     sdata[tid]=g_idata[i];
//     __syncthreads();
//
//     for(unsigned int s=blockDim.x/2;s>0;s>>=1){
//         if(tid<s){
//             sdata[tid] += sdata[tid+s];
//         }
//         __syncthreads();
//     }
//
//     if(tid == 0){ g_odata[blockIdx.x] = sdata[0];}
// }
//
// __global__ void reduce4(int *g_idata, int *g_odata){
//     extern __shared__ int sdata[];
//
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//     sdata[tid]=g_idata[i] + g_idata[i+blockDim.x];
//     __syncthreads();
//
//     for(unsigned int s=blockDim.x/2;s>0;s>>=1){
//         if(tid<s){
//             sdata[tid] += sdata[tid+s];
//         }
//         __syncthreads();
//     }
//
//     if(tid == 0){ g_odata[blockIdx.x] = sdata[0];}
// }
//
// __global__ void reduce5(int *g_idata, int *g_odata){
//     extern __shared__ int sdata[];
//
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//     sdata[tid]=g_idata[i] + g_idata[i+blockDim.x];
//     __syncthreads();
//
//     for(unsigned int s=blockDim.x/2;s>32;s>>=1){
//         if(tid<s){
//             sdata[tid] += sdata[tid+s];
//         }
//         __syncthreads();
//     }
//
//     if(tid == 0){ g_odata[blockIdx.x] = sdata[0];}
// }
//
// __global__ void reduce6(int *g_idata, int *g_odata){
//     extern __shared__ int sdata[];
//
//     unsigned int tid = threadIdx.x;
//     unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
//     sdata[tid]=g_idata[i] + g_idata[i+blockDim.x];
//     __syncthreads();
//
//     for(unsigned int s=blockDim.x/2;s>32;s>>=1){
//         if(tid<s){
//             sdata[tid] += sdata[tid+s];
//         }
//         __syncthreads();
//     }
//
//     if(tid == 0){ g_odata[blockIdx.x] = sdata[0];}
// }

// template
// __global__ void offset(T* a, int s){
//     int i = blockDim.x * blockIdx.x + threadIdx.x + s;
//     a[i]=a[i]+1;
// }
//
// template
// __global__ void stride(T* a,int s){
//     int i=(blockDim.x * blockIdx.x + threadIdx.x)*s;
//     a[i]=a[i]+1;
// }


// __device__ int sumReduction(thread_group tg, int *x, int val){
//
// }

// template <unsigned int blockSize>
// __device__ void warpReduce(volatile int *sdata, int tid){
//     if(blockSize>=64)sdata[tid]+=sdata[tid+32];
//     if(blockSize>=32)sdata[tid]+=sdata[tid+16];
//     if(blockSize>=16)sdata[tid]+=sdata[tid+8];
//     if(blockSize>=8)sdata[tid]+=sdata[tid+4];
//     if(blockSize>=4)sdata[tid]+=sdata[tid+2];
//     if(blockSize>=2)sdata[tid]+=sdata[tid+1];
// }

/**
 * Device kernel to setup pseudo-random number generators that will work with
 * the parallel nature of the GPU thread execution. In order for concurrent
 * threads to generate numbers as intended, there must be a generator
 * initialized for each individual thread.  That is what this kernel does.
 *
 * @method setup_kernel
 * @param  state        A pointer to a certain generator spacific to an
 *                      individual thread id.
 */
// __global__ void setup_kernel(curandState *state){
//     /* Each thread gets same seed, a different sequence num, and no offset */
//     int id = threadIdx.x + blockIdx.x * 64;
//     curand_init(1234,id,0,&state[id]);
// }

/**
 * Kernel to generate the random number sequence based off of the number
 * generators initialized in the setup_kernel.
 *
 * @method generate_kernel
 * @param  state           The pointer to a generator of a spacific thread
 * @param  result          The random number generated w.r.t. the given state.
 */
// __global__ void generate_kernel(curandState *state, int *result){
//     int id = threadIdx.x + blockIdx.x * 64; int count=0;
//     unsigned int x;
//
//     /*Copies state into local mem, to avoid ineficiently accessing global mem*/
//     curandState localState = state[id];
//
//     /*Generation of the pseudorandom numbers*/
//     for(int n=0;n<100000;n++){
//         x=curand(&localState);
//         if(x & 1){ count++; }
//     }
//
//     /*Back to global mem*/
//     state[id] = localState;
//
//     /*Store results*/
//     result[id]+=count;
// }
