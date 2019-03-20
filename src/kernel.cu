// kernel.cu
// Alex Getz

//#include "kernel.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <random>
#include <time.h>
#include <string>
#include <array>
#include <vector>
#include <algorithm>
#define PN 1000
#define sn 100

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

/* ///   HOST FUNCTIONS   /// */
int RanNumber(){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0,1000);
    return dist(mt);
}

//////////////////////////////////////////////////////////////////////////////
/* This function comes from the NVidia Developer Blog entitled:
 * "How to Optimize Data Transfers in CUDA C/C++"
 * link: https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 */
//////////////////////////////////////////////////////////////////////////////
void profileCopies(int *h_sample, int *h_b, int *d,
                   unsigned int num,
                   char *desc)
{
    printf("\n%s transfers\n", desc);

  unsigned int bytes = num * sizeof(int);

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_sample, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (bytes/s): %f\n", bytes / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (bytes/s): %f\n", bytes / time);

  for (int i = 0; i < num; ++i) {
    if (h_sample[i] != h_b[i]) {
      printf("*** %s transfers failed ***\n", desc);
      break;
    }
  }

  // clean up events
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}
//////////////////////////////////////////////////////////////////////////////


/* ///   DEVICE FUNCTIONS   /// */


/* ///   GLOBAL FUNCTIONS   /// */
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


/**MAIN
 *
 *
 */
int main(){
    // Variables
    //unsigned int bytes = sn*sizeof(int);
    unsigned int nElements = 100;
    const unsigned int bytes = nElements * sizeof(int);
    std::vector<int> pData;
    int *BaseSample,/* *h_APinn*/ *d_Base,/* *h_BPage*/ *h_BPinn;
    std::array<int,sn> sKey; //Keeps track of the indexs of sampled pData
    std::string sbuffer;


    // Allocate memory in GPU for the sample data
    //BaseSample = (int*)malloc(bytes);
    checkCuda( cudaMallocHost((int**)&BaseSample,bytes));
    checkCuda( cudaMallocHost((int**)&h_BPinn,bytes));
    //h_BPage = (int*)malloc(bytes);
    //checkCuda( cudaMallocHost((void**)&h_APinn, bytes) ); // host pinned
    //checkCuda( cudaMallocHost((void**)&h_BPinn, bytes) ); // host pinned
    checkCuda( cudaMalloc((void**)&d_Base, bytes) );
    //cudaMemcpy(d_Base,BaseSample,bytes,cudaMemcpyHostToDevice);


    // Read in Population data and store it
    std::ifstream fs("../data/data1.txt");
    if(!fs){std::cerr<<"Cannot open the data file!"<<std::endl;}
    else{
        while(std::getline(fs,sbuffer)){
            pData.push_back(std::atoi(sbuffer.c_str()));
        }
    }
    fs.close();

    // Take a sample of 100 unique elements of population data
	int temp;
    for(int i=0;i<sn;++i){
        int count=0;
        do{
            temp=RanNumber();
            //std::cout<<"i = "<<i<<" :  "<<temp<<std::endl;
            ++count;
        }while(std::any_of(sKey.begin(),sKey.end(),[&](int x){return x==temp;}));
        sKey[i]=temp;
        BaseSample[i]=pData[temp];
        //std::cout<<"BaseSample element "<<i<<" :  "<<BaseSample[i]<<" with key of "<<sKey[i]<<std::endl;
        if(count>1){
            printf("While loop pass count = ");
            std::cout<<count<<" on element i = "<<i<<std::endl;
        }
    }


    // Officially set
    // printf("HERE 1\n");
    // memcpy(h_APinn,BaseSample,bytes);
    // printf("HERE 2\n");
    // memset(h_BPage,0,bytes);
    // printf("HERE 3\n");
    // memset(h_BPinn,0,bytes);
    // printf("HERE 4\n");


    // output device info and transfer size
    // cudaDeviceProp prop;
    // checkCuda( cudaGetDeviceProperties(&prop, 0) );
    // printf("\nDevice: %s\n", prop.name);
    // printf("Transfer size (Bytes): %d\n", bytes);

    // perform copies and report bandwidth
    //profileCopies(BaseSample, h_BPinn, d_Base, nElements, "Pageable");
    profileCopies(BaseSample, h_BPinn, d_Base, nElements, "Pinned");
    printf("n");

    // Finished operations and now returning data from device to host
    cudaFree(d_Base);
    cudaFreeHost(BaseSample);
    cudaFreeHost(h_BPinn);
    // cudaFreeHost(h_BPage);
    // free(h_APinn);
    // free(h_BPage);
    return 0;
}

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
