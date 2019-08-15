// kernel.cu
// Alex Getz

#include <fstream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <array>
#include <assert.h>

#define MAX_ENTRIES 11897026
#define B_SIZE 2000
#define TPB 128

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


/* ///   DEVICE FUNCTIONS   /// */
__device__ float getnextrand(curandState *state){
    return (float)(curand_uniform(state));
}

__device__ int getnextrandscaled(curandState *state, unsigned long int scale){
    return (unsigned long int) scale * getnextrand(state);
}

__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void bootstrap(int *output_mean, int *d_sample, curandState *state)
{
    /*  */
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned long int ts;
    long long int sum = 0;
    for(int i=0;i<MAX_ENTRIES;++i){
        ts = getnextrandscaled(&state[idx], MAX_ENTRIES);
        sum += d_sample[ts];
    }
    output_mean[idx] = (sum/MAX_ENTRIES);
}

int main(){

    int *BaseSample, *d_Base;
    int *d_mean, *h_mean;
    curandState *devStates;
    checkCuda( cudaMallocHost((void**)&BaseSample,MAX_ENTRIES*sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&devStates,2000*sizeof(curandState)));

    // int *pData_h;
    std::string line;

    // pData_h = (int*)malloc(MAX_ENTRIES*sizeof(int));

    uintmax_t m_numLines = 0;
    std::ifstream fs("../data/allCountries.txt");
    if(!fs){
        /*std::cerr */
        std::cout<<"ERROR\n";
    }else{
        while (std::getline(fs, line))
        {
            int counter=0;
            std::stringstream ss;
            std::string temp;
            // std::cout<<"\n"<<line<<"\n";
            ss << line;
            std::getline(ss,temp,'\t');
            // std::cout<<temp<<", position: "<<++counter<<"\n";
            while(std::getline(ss,temp,'\t')){
                if(temp.length() == 4){
                    BaseSample[m_numLines] = std::atoi(temp.c_str());



                    // std::cout<<temp<<", position: "<<++counter<<"\n";
                    break;
                } else{ ++counter; }
            }
            m_numLines++;
            // if(m_numLines==5){ break; }
        }
    }
    std::cout << "m_numLines = " << m_numLines << "\nMoving on...\n\n";
    fs.close();

    //std::cout << "Element 300,000 of BaseSample: " << BaseSample[300000]<<std::endl;


    ///////////////////////////////////////////////////////////////////////////
    checkCuda( cudaMalloc((void**)&d_Base,MAX_ENTRIES*sizeof(int)));
    checkCuda( cudaMemcpy(d_Base,BaseSample,MAX_ENTRIES*sizeof(int),cudaMemcpyHostToDevice));
    cudaFreeHost(BaseSample);

    checkCuda( cudaMalloc((void**)&d_mean,2000*sizeof(int)));
    checkCuda( cudaMallocHost((void**)&h_mean,2000*sizeof(int)));
    // 2048 bootstraps of 128 threads each

    //////////////////////////////////////
    initCurand<<<(2000+128-1)/128,128>>>(devStates, 1234);
    cudaDeviceSynchronize();
    bootstrap<<<(2000+128-1)/128,128>>>(d_mean, d_Base, devStates);
    cudaDeviceSynchronize();

    checkCuda( cudaMemcpy(h_mean,d_mean,2000*sizeof(int),cudaMemcpyDeviceToHost));
    for(int i=0;i<2000;++i){
        std::cout<<"element "<<i<<" : "<<h_mean[i]<<std::endl;
    }

    cudaFree(d_Base);
    cudaFree(d_mean);

    printf("\n\n\nDONE\n\n\n");
    cudaFree(devStates);
    cudaFreeHost(h_mean);


    return 0;
}


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
