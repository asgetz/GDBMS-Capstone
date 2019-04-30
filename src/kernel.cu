// kernel.cu
// Alex Getz

// #include <cuda_runtime_api.h>
// #include <device_launch_parameters.h>
// #include <cooperative_groups.h>
// #include <crt/device_functions.h>
// #include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <KernelInterface.h>
#include <assert.h>

// #include <algorithm>
// #include <iostream>
// #include <sstream>
// #include <cstring>
// #include <string>
// #include <vector>

// #include <random>
// #include <time.h>
// #include <math.h>
// #include <array>


// inline
// cudaError_t checkCuda(cudaError_t result)
// {
// #if defined(DEBUG) || defined(_DEBUG)
//     if(result!=cudaSuccess){
//         fprintf(stderr,"CUDA Runtime Error: %s\n",
//                 cudaGetErrorString(result));
//         assert(result==cudaSuccess);
//     }
// #endif
//     return result;
// }
//
// #define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
//     printf("Error at %s:%d\n",__FILE__,__LINE__); \
//     return EXIT_FAILURE;}} while(0)


// #define MUL(a, b) a*b
//
//
// /* ///   Host Functions     /// */
//
//
// /* ///   DEVICE FUNCTIONS   /// */
//
//
// /* ///   GLOBAL FUNCTIONS   /// */
// /**
//  * [addKernel description]
//  * @method addKernel
//  * @param  c         [description]
//  * @param  a         [description]
//  * @param  b         [description]
//  */
// __global__ void addKernel(int *c, const int *a, const int *b)
// {
//     int i = threadIdx.x;
//     c[i] = a[i] + b[i];
// }
//
//
// /**
//  * CUDA Kernel designated to carry out the bootstrap of the Base sample.
//  * When launched, this kernel will generate a random number generator for
//  * each thread that executes with it. These threads, with their random
//  * number, will choose a random element from the original sample. Then the
//  * kernel will construct an array of x_bar values by summation amongst the
//  * threads.
//  * @method bootstrap
//  * @param  out       This is the output array, it will contain the x_bar values
//  * @param  d_sample  This is the original sample that gets copied onto the GPU
//  * @param  state     An array of pointers to unique cuRAND instances for threads
//  * @return           return.
//  */
// __global__ void bootstrap(unsigned int *out, int *d_sample)
// {
//     /*  */
//     // unsigned int tid = threadIdx.x;
//     // unsigned int block = blockIdx.x;
//     // unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
//     // unsigned int ts;
//     //
//     // curandState localState = state[idx];
//     // // curand_init(1234,0,idx,&localState);
//     // // __syncthreads();
//     //
//     //
//     // ts = curand(&localState);
//     // out[idx]=d_sample[ts%100];
//     // __syncthreads();
//
//     /*
//      * Attempt to separate the blocks of 128 threads into
//      * logical groupings of 100 for summation
//      */
//     // int x = idx%100;
//     // int tx = idx-x;
//     // int y = idx%128;
//     // int ty = idx-y;
//     // if(ty>tx){
//     //     out[idx]=99999;
//     // }else{
//     //     out[idx]=(idx%100)+1;
//     // }
// }
//
//
// /**
//  * [parkmillerKernel description]
//  * @method parkmillerKernel
//  * @param  d_Output         [description]
//  * @param  seed             [description]
//  * @param  cycles           [description]
//  * @param  N                [description]
//  */
// static __global__ void parkmillerKernel(int *d_array, unsigned int seed, \
//                                         int cycles, unsigned int N)
// {
//     unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
//     unsigned int  threadN = MUL(blockDim.x, gridDim.x);
//     double const a    = 16807;      //ie 7**5
//     double const m    = 2147483647; //ie 2**31-1
//     double const reciprocal_m = 1.0/m;
//
//     for(unsigned int pos = tid; pos < N; pos += threadN){
//         unsigned int result = 0;
//         unsigned int data = seed + pos;
//
//         for(int i = 0; i < cycles; i++) {
//             double temp = data * a;
//             result = (int) (temp - m * floor ( temp * reciprocal_m ));
//             data = result;
//         } //endfor
//
//         //d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
//         //d_Output[MUL(threadIdx.y, N) + pos] = result;
//         d_array[pos] = result;
//     }
// }












// /**/
// void addKernelCall(unsigned int bnum, unsigned int tnum, int *c, \
//                    const int *a, const int *b)
// {
//     /**/
//     //
//     // addKernel<<<bnum,tnum>>>(c,a,b);
// }
//
// /**/
// void bootstrapCall(unsigned int bnum, unsigned int tnum, \
//                   unsigned int *output, int *sampleInput)
// {
//     /**/
//     // bootstrap<<<bnum,tnum>>>(output, sampleInput);
// }
//
// /**/
// void parkmillerCall(int *Out, unsigned int seed, int cycles, \
//                    unsigned int grid_size, unsigned int block_size, \
//                    unsigned int N)
// {
//     /**/
//     // parkmillerKernel<<<grid_size, block_size>>>(Out,seed,cycles,N);
// }



// // #define MAX_ENTRIES 11897027
//
// /**
//  * [main description]
//  * @method main
//  * @param  int  [description]
//  * @return      [description]
//  */
// int main(){
//
//     /* Variables have been offloaded into the KernelInterface class within
//      * kernel.h
//      * Most are public and/or generated by the class constructor
//      * So then create an opject to reference it by here.
//      */
//     //
//
//     // KernelInterface *obj;
//
//     //
//     //////// Read in Population data and store it
//     // using boost::iostreams::mapped_file_source;
//     // using boost::iostreams::stream;
//     // mapped_file_source mmap("../data/allCountries.txt");
//     // stream<mapped_file_source> is(mmap, std::ios::binary);
//     //
//     // int *pData_h;
//     // std::string line;
//     //
//     // uintmax_t m_numLines = 0;
//     // while (std::getline(is, line))
//     // {
//     //     int counter=0;
//     //     std::stringstream ss;
//     //     std::string temp;
//     //     std::cout<<"\n"<<line<<"\n";
//     //     ss << line;
//     //     std::getline(ss,temp,'\t');
//     //     // std::cout<<temp<<", position: "<<++counter<<"\n";
//     //     while(std::getline(ss,temp,'\t')){
//     //         if(temp.length() == 4){
//     //
//     //             // Right now treat whole input stream as the sample
//     //             // Later will add ability to distinguish what size of sample you want.
//     //             // numEntries[i]=std::atoi(entries[i].c_str());
//     //             pData_h[m_numLines] = std::atoi(temp.c_str());
//     //
//     //
//     //
//     //             // std::cout<<temp<<", position: "<<++counter<<"\n";
//     //             break;
//     //         } else{ ++counter; }
//     //     }
//     //     m_numLines++;
//     //     // if(m_numLines==5){ break; }
//     // }
//     // std::cout << "m_numLines = " << m_numLines << "\n";
//
//     //////////////////// Copy to Device, launch, and copy back to host
//     /* All of these steps have been offloaded onto the KernelInterface class */
//
//     //////////////////// Statistical analysis ; WILL OFFLOAD
//     /* At the very least calculate the pertinent bootstrap things
//      * Maybe even go further with histograms and charts through nvvp or something
//      */
//     //
//
//
//
//     //////////////////// Free veriables. Possibly invoke class deconstructor
//
//     // printf("\n\n\nDONE\n\n\n");
//     return 0;
// }


//////// Take a sample of 100 unique elements of population data
// int temp;
// int sumOriginal = 0;
// int meanOriginal;
// for(int i=0;i<SAMPLE_SIZE;++i){
//     int count=0;
//     do{
//         temp=PopRand();
//         //std::cout<<"i = "<<i<<" :  "<<temp<<std::endl;
//         ++count;
//     }while(std::any_of(sKey.begin(),sKey.end(),[&](int x){return x==temp;}));
//     sKey[i]=temp;
//     BaseSample[i]=pData[temp];
//     sumOriginal+=BaseSample[i];
//     /////////////// IF STATEMENT PRINTF TO CHECK IF THE OUTPUT IS CORRECT
//     //std::cout<<"BaseSample element "<<i<<" :  "<<BaseSample[i]<<" with key of "<<sKey[i]<<std::endl;
//     if(count>1){
//         printf("While loop pass count = ");
//         std::cout<<count<<" on element i = "<<i<<std::endl;
//     }
// }
// meanOriginal = sumOriginal/100;


//////////////////// Calculate Statistics. Soon to be offloaded
// int finalMean[400]={0};
// // std::vector<int> *meanVector;
// int bnum = 0;
// int sum1;
// int sum2 = 0;
// // int temp1;
// for (int a=0;a<400;++a){
//     sum1=0;
//     for(int b=0;b<100;++b){
//         sum1+=h_mean[b+(100*bnum)];
//     }
//     finalMean[a]=sum1/100;
//     // temp1 = sum1/100;
//     // meanVector[a].push_back( temp1 );
//     sum2 += std::pow( (finalMean[a]-meanOriginal), 2 );
//     bnum++;
//     // std::cout<<"Final Mean "<<a<<" : "<<finalMean[a]<<std::endl;
// }
// printf("\n\n\n");
// std::sort(finalMean,finalMean+SAMPLE_SIZE);
// std::cout<<"sum2 is "<<sum2<<std::endl;
// int div = 400;
// std::cout<<"div is "<<div<<std::endl;
// float stdDeviation = sqrt( (sum2/div) );
// std::cout<<"Standard Deviation is "<<stdDeviation<<std::endl;
// float stdErrorFactor = ( 100.0 / (100.0-1.0) );
// std::cout<<"The Error Factor is "<<stdErrorFactor<<std::endl;
// float stdError = sqrt( stdErrorFactor ) * stdDeviation;
// std::cout<<"Standard Error is "<<stdError<<std::endl;
// int tempA; int tempB;
// float lowerCI = 400 * ( 0.05/2 );
// tempA = finalMean[(int)lowerCI];
// std::cout<<"Lower (5%) Confidence Interval is "<<tempA<<std::endl;
// float higherCI = 400 * ( 1 - (0.05/2) );
// tempB = finalMean[(int)higherCI];
// std::cout<<"Higher (95%) Confidence Interval is "<<tempB<<std::endl;
//
// //////// Free data allocation from Device and Host
// cudaFree(d_Base);
// cudaFree(d_mean);
// cudaFreeHost(BaseSample);
// cudaFreeHost(h_mean);
