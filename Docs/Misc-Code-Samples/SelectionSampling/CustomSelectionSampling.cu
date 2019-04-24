#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <fstream>
// #include <boost/iostreams/stream.hpp>
// #include <libs/iostreams/src/mapped_file.cpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <ctime>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>

#define MAX_ENTRIES 11897027

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

///////////////////////////////////////////////////////////////////////////////

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



__host__ __device__ class Point2D
{
// public:
// 	float x, y, w;
// 	__host__ __device__ Point2D& operator=(const Point2D& target){ x = target.x; y = target.y; w = target.w; return *this; }
// 	__host__ __device__ Point2D operator+(const Point2D& b){ Point2D results; results.x = x + b.x; results.y = y + b.y; results.w = w + b.w; return results; }
// 	__host__ __device__ Point2D operator+(const float b)
// 	{
// 		Point2D results;
// 		results.x = min_(1, x + b);
// 		results.y = min_(1, y + b);
// 		return results;
// 	}
// 	__host__ __device__ Point2D operator-(const float b)
// 	{
// 		Point2D results;
// 		results.x = max_(0,x - b);
// 		results.y = max_(0,y - b);
// 		return results;
// 	}
// 	friend ostream& operator<<(ostream& os, const Point2D& p)
// 	{
// 		os << p.x<<" " <<p.y<<" "<<p.w ;
// 		return os;
// 	}
};

//Generate a million random points;

void generateRandomPointCloud(vector<Point2D>& points, size_t size = 1000000)
{
	//std::cout << "Generating " << size << " point cloud...";
	points.resize(size);
	for (size_t i = 0; i<size; i++)
	{

		points[i].x = (rand() % RAND_MAX) / float(RAND_MAX);
		points[i].y = (rand() % RAND_MAX) / float(RAND_MAX);
		points[i].w = 0.0;
	}

	//std::cout << "done\n";
}





int int main(int argc, char const *argv[]) {
    /* code */
    return 0;
}





// #define GPUCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
// {
// 	if (code != cudaSuccess)
// 	{
// 		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
// 		if (abort) exit(code);
// 	}
// }
//
// cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
// __global__ void addKernel(int *c, const int *a, const int *b)
// {
//     int i = threadIdx.x;
//     c[i] = a[i] + b[i];
// }
