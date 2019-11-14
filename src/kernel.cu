// kernel.cu
// Alex Getz

#include <cstddef>
#include <stdexcept>
#include <memory>
#include <cuda_profiler_api.h>

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

#include <random>
#include <functional>
#include <math.h>
#include <algorithm>
#include <array>

#define MAX_ENTRIES 11897026
#define B_SIZE 2000
#define TPB 128
#define SHMEM_SIZE 1024 * sizeof(int)








///////////////////////////////////////////////////////////////////////////////////////
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
    return EXIT_FAILURE;}} /*while(0)*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if(abort) exit(code);
	}
}
////////////////////////////////////////////////////////////////////////////////////////





/* ///   DEVICE FUNCTIONS   /// */

__device__ float getnextrand(curandState *state){
    return (float)(curand_uniform(state));
}

__device__ int getnextrandscaled(curandState *state, unsigned long int scale){
    return (unsigned long int) scale * getnextrand(state);
}


/* ///   DEVICE KERNELS   /// */

__global__ void initCurand(curandState *state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(idx+seed, 0, 0, &state[idx]);
}

__global__ void bootstrap(double *output_mean, int *d_sample, curandState *state)
{
    __shared__ int partial_Sums[SHMEM_SIZE];

    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tNum = threadIdx.x;
    int bNum = blockIdx.x;
    int bSize = blockDim.x;
    unsigned long int ts = 0;
    long long int tSum = 0;
    int count = 0;

    for(unsigned int i=tNum; i<MAX_ENTRIES; i+=bSize){
	ts = getnextrandscaled(&state[idx], MAX_ENTRIES);
	tSum += d_sample[ts];
	count++;
    }
    partial_Sums[tNum] = tSum / count;
    __syncthreads();

    // Perform Sum reduction across all the threads of the block
    for(int s=(bSize/2); s>0; s >>= 1){
	// Each thread does work unless the index goes off the block
	if(tNum<s){
	    partial_Sums[tNum] += partial_Sums[tNum+s];
	}
	__syncthreads();
    }

    // Use first thread of each block to write results back to main mem
    if(tNum==0){
        output_mean[bNum] = (double)partial_Sums[0] / (double)bSize;
    }
}



/* ///   HOST GLOBAL FUNCTIONS   /// */

void throw_error(cudaError_t err){
    if(err != cudaSuccess)
	throw std::runtime_error(cudaGetErrorString(err));
}


/* ///   HOST GLOBAL STRUCTS & VARIABLES   /// */

struct cuda_free_deleter_t{
    void operator()(void* ptr) const
    {
	cudaFree(ptr);
    }
};


template <typename T>
auto cudaAllocBuffer(std::size_t size){
    void *ptr;
    throw_error(cudaMalloc(&ptr, size*sizeof(T)));
    return std::unique_ptr<T, cuda_free_deleter_t> { static_cast<T*>(ptr) };
}


int main(){
    int *BaseSample, *d_Base;
    double *d_mean, *h_mean;
    //curandState *devStates;
    checkCuda( cudaMallocHost((void**)&BaseSample,MAX_ENTRIES*sizeof(int)));
    //checkCuda( cudaMalloc((void**)&devStates,2048*1024*sizeof(curandState)));

    std::string line;
    uintmax_t m_numLines = 0;
    std::ifstream fs("../data/allCountries.txt");
    if(!fs){
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
    //std::cout << "m_numLines = " << m_numLines << "\nMoving on...\n\n";
    fs.close();

    //std::cout << "Element 300,000 of BaseSample: " << BaseSample[300000]<<std::endl;


    ///---			Calculating the Mean			 ---///
    ///////////////////////////////////////////////////////////////////////////
    std::cout<<"Sample has been generated.\nCalculating the mean...\n";
    long long int BaseSum = 0;
    for(int i=0;i<MAX_ENTRIES;i++){
	BaseSum += BaseSample[i];
    }
    double BaseMean = (double)BaseSum / (double)MAX_ENTRIES;
    std::cout<<"Mean has been calculated! Moving on...\n\n";
    


    ///---			KERNEL OPERATIONS			 ---///
    ///////////////////////////////////////////////////////////////////////////
    checkCuda( cudaMalloc((void**)&d_Base,MAX_ENTRIES*sizeof(int)));
    checkCuda( cudaMemcpy(d_Base,BaseSample,MAX_ENTRIES*sizeof(int),cudaMemcpyHostToDevice));


    //checkCuda( cudaFreeHost(BaseSample) );

    checkCuda( cudaMalloc((void**)&d_mean,2048*sizeof(double)));
    checkCuda( cudaMallocHost((void**)&h_mean,2048*sizeof(double)));

    std::cout<<"Launching initCurand Kernel now\n\n";

    //////////////////////////////////////
    //checkCuda( cudaProfilerStart() );
    try{
	constexpr int block_size = 512;
	constexpr int num_blocks = 4096;
	auto devStates = cudaAllocBuffer<curandState>(num_blocks * block_size);
	initCurand<<<num_blocks, block_size>>>(devStates.get(),1234);
	throw_error(cudaPeekAtLastError());
	throw_error(cudaDeviceSynchronize());
	std::cout<<"Curand Kernel Launch Try block SUCCESSFUL!\n";
	std::cout<<"Launching Bootstrap Kernel now\n\n";
	bootstrap<<<2048,1024>>>(d_mean,d_Base,devStates.get());
	throw_error(cudaPeekAtLastError());
	throw_error(cudaDeviceSynchronize());
	std::cout<<"Bootstrap Kernel Launch Try Block SUCCESSFUL!\n";
    }
    catch (const std::exception& e)
    {
	std::cerr << "Error: " << e.what() << '\n';
	return -1;
    }
    catch (...)
    {
	std::cerr << "Unknown Exception";
	return -1;
    }

    std::cout<<"Kernels appear complete, attempting to copy data back to Host\n";
    checkCuda( cudaMemcpy(h_mean,d_mean,2048*sizeof(double),cudaMemcpyDeviceToHost) );

    /* This loop is meant for testing the validity of the memcpy output
    for(int i=0;i<2048;++i){
        std::cout<<"element "<<i<<" : "<<h_mean[i]<<std::endl;
    }
    */

    // Standard Error of the bootstrap means
    int n = 2048;
    double SumOfMeans=0;
    for(int i=0;i<n;i++){
	SumOfMeans+=h_mean[i];
    }
    double MeanOfMeans = SumOfMeans / (double)n;
    std::cout<<"MeanofMeans: "<<MeanOfMeans<<"\n";

    double SqrDiff=0;
    for(int i=0;i<n;i++){
	SqrDiff += (h_mean[i]-MeanOfMeans) * (h_mean[i]-MeanOfMeans);
    }
    double SqrdVariance = SqrDiff / (n-1);
    double BootError = ((n-1)/(n*n)) * SqrdVariance;
    std::cout<<"SqrDiff, SqrdVariance, & BootError: "<<SqrDiff<<", "<<SqrdVariance<<", "<<BootError<<"\n";

    //std::sort(h_mean,h_mean+2048);
    std::cout<<"\nStandard Error is: "<<BootError<<"\n\n\n";

    double C_Arr[2048] = {};
    for(int i=0;i<n;i++){
	C_Arr[i]=h_mean[i]-BaseMean;
    }
    std::sort(std::begin(C_Arr),std::end(C_Arr));
    double L = 2048.0 * 0.1;
    int Lower = (int)L;
    double H = 2048 * 0.9;
    int Higher = (int)H;

    int LowerBound = C_Arr[Lower];
    int UpperBound = C_Arr[Higher];

    double Left = BaseMean - (double)LowerBound;
    double Right = BaseMean - (double)UpperBound;

    std::cout<<"\n\n\n------------------------------------------------\n";
    std::cout<<"The Confidence Interval is: 80%\n";
    std::cout<<"The Standard Error is: "<<BootError<<"\n";
    std::cout<<"This is on the 10th & 90th percentiles: ["<<Left<<", "<<Right<<"]\n";

    checkCuda( cudaFree(d_Base) );
    checkCuda( cudaFree(d_mean) );
    //checkCuda( cudaFree(devStates) );
    checkCuda( cudaFreeHost(BaseSample) );
    checkCuda( cudaFreeHost(h_mean) );
    printf("\n\n\nDONE\n");


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
