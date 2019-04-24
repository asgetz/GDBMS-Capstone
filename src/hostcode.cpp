// kernel.cu
// Alex Getz

#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "kernel.cu"

#include <fstream>
#include <boost/iostreams/stream.hpp>
#include <libs/iostreams/src/mapped_file.cpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <assert.h>
#include <random>
#include <time.h>
#include <math.h>
#include <array>

// #include <thrust/execution_policy.h>
// #include <thrust/device_vector.h>
// #include <thrust/transform.h>
// #include <thrust/sequence.h>
// #include <thrust/copy.h>
// #include <thrust/fill.h>
// #include <thrust/replace.h>
// #include <thrust/functional.h>
// #include <thrust/scan.h>
#include <ctime>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

#define MAX_ENTRIES 11897027
// #define POP_SIZE 1000
// #define SAMPLE_SIZE 100

// namespace cg = cooperative_groups;

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






// extern "C" void parkmiller_gpu(int *d_Output, unsigned int seed, int cycles,
// 			       unsigned int grid_size,unsigned int block_size, unsigned int N);


/**
 * [main description]
 * @method main
 * @param  int  [description]
 * @return      [description]
 */
int main(){
    // // Variables
    // const unsigned int bytes = SAMPLE_SIZE * sizeof(int);
    // // const int d_rand_size = 40000;
    // // const unsigned int d_rand_bytes = d_rand_size * sizeof(int);
    // std::vector<int> pData;
    // // int d_rand[d_rand_size];
    // int *BaseSample, *d_Base/*, *d_rand*/, *h_mean;
    // unsigned int * d_mean;
    // std::array<int,SAMPLE_SIZE> sKey; //Keeps track of the indexs of sampled pData
    // std::string sbuffer;
    //
    // unsigned int total;
    // curandState *devStates;

    /////// Allocate space for prng states on device
    // CUDA_CALL(cudaMalloc((void**)&devStates, 313*128*sizeof(curandState)));


    //////// Allocate memory in GPU for the sample data
    // checkCuda( cudaMallocHost((int**)&BaseSample,bytes));
    // cudaMalloc((void**)&d_Base,bytes);
    // checkCuda( cudaMallocHost((void**)&h_mean,313*128*sizeof(int)));
    // checkCuda( cudaMalloc((void**)&d_mean,313*128*sizeof(int)));


    //////// Read in Population data and store it
    using boost::iostreams::mapped_file_source;
    using boost::iostreams::stream;
    mapped_file_source mmap("../data/allCountries.txt");
    stream<mapped_file_source> is(mmap, std::ios::binary);

    std::vector<int> pData;

    std::string line;

    uintmax_t m_numLines = 0;
    while (std::getline(is, line))
    {
        int counter=0;
        std::stringstream ss;
        std::string temp;
        std::cout<<"\n"<<line<<"\n";
        ss << line;
        std::getline(ss,temp,'\t');
        std::cout<<temp<<", position: "<<++counter<<"\n";
        while(std::getline(ss,temp,'\t')){
            if(temp.length() == 4){
                std::cout<<temp<<", position: "<<++counter<<"\n";
                break;
            } else{ ++counter; }
        }
        m_numLines++;
        if(m_numLines==5){ break; }
    }
    std::cout << "m_numLines = " << m_numLines << "\n";

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

    // For testing
    // std::cout<<"\n\n\nBASE RANDOM SAMPLE GENERATED\nPress enter to continue...";
    // std::cin.ignore();
    // printf("\nMOVING ON...\n");
    //
    // //////// Copy from Host to device
    // cudaMemcpy(d_Base,BaseSample,bytes,cudaMemcpyHostToDevice);
    // // cudaMemcpy(d_mean,h_mean,bytes,cudaMemcpyHostToDevice);
    //
    // //////// Kernel Launch
    // bootstrap<<<313,128>>>(d_mean, d_Base, devStates);
    //
    // //////// Copy back from device to host
    // cudaMemcpy(h_mean,d_mean,313*128*sizeof(int),cudaMemcpyDeviceToHost);
    // // for(int c=0;c<313*128;++c){
    // //     std::cout<<"element "<<c<<" : "<<h_mean[c]<<std::endl;
    // // }
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

    printf("\n\n\nDONE\n\n\n");
    return 0;
}
