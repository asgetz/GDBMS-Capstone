// kernel.cu
// Alex Getz

//#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <time.h>
#include <math.h>
#include <string>
#include <array>
#include <vector>
#include <algorithm>
#define POP_SIZE 1000
#define SAMPLE_SIZE 100

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



/* ///   HOST FUNCTIONS   /// */
int PopRand(){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> popdist(0,1000);
    return popdist(mt);
}

int SampleRand(){
    std::random_device sample_rd;
    std::mt19937 sample_mt(sample_rd());
    std::uniform_int_distribution<int> sampledist(0,100);
    return sampledist(sample_mt);
}

/* ///   DEVICE FUNCTIONS   /// */


/* ///   GLOBAL FUNCTIONS   /// */
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
__global__
void bootstrap(unsigned int *out, int *d_sample, curandState *state){
    /*  */
    unsigned int tid = threadIdx.x;
    unsigned int block = blockIdx.x;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int ts;

    curandState localState = state[idx];
    curand_init(1234,0,idx,&localState);
    // __syncthreads();


    ts = curand(&localState);
    out[idx]=d_sample[ts%100];
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
 * [main description]
 * @method main
 * @param  int  [description]
 * @return      [description]
 */
int main(int){
    // Variables
    const unsigned int bytes = SAMPLE_SIZE * sizeof(int);
    // const int d_rand_size = 40000;
    // const unsigned int d_rand_bytes = d_rand_size * sizeof(int);
    std::vector<int> pData;
    // int d_rand[d_rand_size];
    int *BaseSample, *d_Base/*, *d_rand*/, *h_mean;
    unsigned int * d_mean;
    std::array<int,SAMPLE_SIZE> sKey; //Keeps track of the indexs of sampled pData
    std::string sbuffer;

    unsigned int total;
    curandState *devStates;

    /////// Allocate space for prng states on device
    CUDA_CALL(cudaMalloc((void**)&devStates, 313*128*sizeof(curandState)));


    //////// Allocate memory in GPU for the sample data
    checkCuda( cudaMallocHost((int**)&BaseSample,bytes));
    cudaMalloc((void**)&d_Base,bytes);
    checkCuda( cudaMallocHost((void**)&h_mean,313*128*sizeof(int)));
    checkCuda( cudaMalloc((void**)&d_mean,313*128*sizeof(int)));


    //////// Read in Population data and store it
    std::ifstream fs("../data/data1.txt");
    if(!fs){std::cerr<<"Cannot open the data file!"<<std::endl;}
    else{
        while(std::getline(fs,sbuffer)){
            pData.push_back(std::atoi(sbuffer.c_str()));
        }
    }
    fs.close();

    //////// Take a sample of 100 unique elements of population data
	int temp;
    int sumOriginal = 0;
    int meanOriginal;
    for(int i=0;i<SAMPLE_SIZE;++i){
        int count=0;
        do{
            temp=PopRand();
            //std::cout<<"i = "<<i<<" :  "<<temp<<std::endl;
            ++count;
        }while(std::any_of(sKey.begin(),sKey.end(),[&](int x){return x==temp;}));
        sKey[i]=temp;
        BaseSample[i]=pData[temp];
        sumOriginal+=BaseSample[i];
        /////////////// IF STATEMENT PRINTF TO CHECK IF THE OUTPUT IS CORRECT
        //std::cout<<"BaseSample element "<<i<<" :  "<<BaseSample[i]<<" with key of "<<sKey[i]<<std::endl;
        if(count>1){
            printf("While loop pass count = ");
            std::cout<<count<<" on element i = "<<i<<std::endl;
        }
    }
    meanOriginal = sumOriginal/100;

    // For testing
    std::cout<<"\n\n\nBASE RANDOM SAMPLE GENERATED\nPress enter to continue...";
    std::cin.ignore();
    printf("\nMOVING ON...\n");

    //////// Copy from Host to device
    cudaMemcpy(d_Base,BaseSample,bytes,cudaMemcpyHostToDevice);
    // cudaMemcpy(d_mean,h_mean,bytes,cudaMemcpyHostToDevice);

    //////// Kernel Launch
    bootstrap<<<313,128>>>(d_mean, d_Base, devStates);

    //////// Copy back from device to host
    cudaMemcpy(h_mean,d_mean,313*128*sizeof(int),cudaMemcpyDeviceToHost);
    // for(int c=0;c<313*128;++c){
    //     std::cout<<"element "<<c<<" : "<<h_mean[c]<<std::endl;
    // }
    int finalMean[400]={0};
    // std::vector<int> *meanVector;
    int bnum = 0;
    int sum1;
    int sum2 = 0;
    // int temp1;
    for (int a=0;a<400;++a){
        sum1=0;
        for(int b=0;b<100;++b){
            sum1+=h_mean[b+(100*bnum)];
        }
        finalMean[a]=sum1/100;
        // temp1 = sum1/100;
        // meanVector[a].push_back( temp1 );
        sum2 += std::pow( (finalMean[a]-meanOriginal), 2 );
        bnum++;
        // std::cout<<"Final Mean "<<a<<" : "<<finalMean[a]<<std::endl;
    }
    printf("\n\n\n");
    std::sort(finalMean,finalMean+SAMPLE_SIZE);
    std::cout<<"sum2 is "<<sum2<<std::endl;
    int div = 400;
    std::cout<<"div is "<<div<<std::endl;
    float stdDeviation = sqrt( (sum2/div) );
    std::cout<<"Standard Deviation is "<<stdDeviation<<std::endl;
    float stdErrorFactor = ( 100.0 / (100.0-1.0) );
    std::cout<<"The Error Factor is "<<stdErrorFactor<<std::endl;
    float stdError = sqrt( stdErrorFactor ) * stdDeviation;
    std::cout<<"Standard Error is "<<stdError<<std::endl;
    int tempA; int tempB;
    float lowerCI = 400 * ( 0.05/2 );
    tempA = finalMean[(int)lowerCI];
    std::cout<<"Lower (5%) Confidence Interval is "<<tempA<<std::endl;
    float higherCI = 400 * ( 1 - (0.05/2) );
    tempB = finalMean[(int)higherCI];
    std::cout<<"Higher (95%) Confidence Interval is "<<tempB<<std::endl;

    //////// Free data allocation from Device and Host
    cudaFree(d_Base);
    cudaFree(d_mean);
    cudaFreeHost(BaseSample);
    cudaFreeHost(h_mean);

    printf("\n\n\nDONE\n\n\n");
    return 0;
}


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
