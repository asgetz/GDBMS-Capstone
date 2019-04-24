#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
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

//curand
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
#define GPUCHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//heapify subroutines.
__host__ __device__ inline int parent(int i)
{
	return i / 2;
}

__host__ __device__ inline int left(int i)
{
	return 2 * i + 1;
}

__host__ __device__ inline int right(int i)
{
	return 2 * i + 2;
}


template <typename T>
void viewGPUArray(T* array,  int num, string filename)
{
	const string dir = "D:/git/GDBMS-Capstone/Docs/Misc-Code-Samples/Selection Sampling/";
	T* host_array = new T[num];
	GPUCHECK(cudaMemcpy(host_array, array, sizeof(T) * num, cudaMemcpyDeviceToHost));
	ofstream file(dir + filename);
	for (int i = 0; i < num; i++)
	{
		file << host_array[i] << endl;
	}
	file.close();
	delete[] host_array;
}

template <typename T>
void viewGPUArrayMat(T* array, int num, string filename)
{
	const string dir = "D:/git/GDBMS-Capstone/Docs/Misc-Code-Samples/Selection Sampling/";
	T* host_array = new T[num];
	GPUCHECK(cudaMemcpy(host_array, array, sizeof(T) * num, cudaMemcpyDeviceToHost));
	ofstream file(dir + filename);
	for (int i = 0; i < num; i++)
	{
		file << host_array[i].p.x <<" "<<host_array[i].p.y << endl;
	}
	file.close();
	delete[] host_array;
}


//Code on testing hash grid on GPU
//1. Build a hash grid on GPU(Hash grid of a structure I defined)

//The class I'm using
__host__ __device__ inline float max_(const float &a, const float &b)
{
	return ((a > b) ? a : b);
}
__host__ __device__ inline float min_(const float &a, const float &b)
{
	return ((a < b) ? a : b);
}
__host__ __device__ class Point2D
{

public:
	float x, y, w;
	__host__ __device__ Point2D& operator=(const Point2D& target){ x = target.x; y = target.y; w = target.w; return *this; }
	__host__ __device__ Point2D operator+(const Point2D& b){ Point2D results; results.x = x + b.x; results.y = y + b.y; results.w = w + b.w; return results; }
	__host__ __device__ Point2D operator+(const float b)
	{
		Point2D results;
		results.x = min_(1, x + b);
		results.y = min_(1, y + b);
		return results;
	}
	__host__ __device__ Point2D operator-(const float b)
	{
		Point2D results;
		results.x = max_(0,x - b);
		results.y = max_(0,y - b);
		return results;
	}
	friend ostream& operator<<(ostream& os, const Point2D& p)
	{
		os << p.x<<" " <<p.y<<" "<<p.w ;
		return os;
	}
};
struct BBox
{
	float minx, miny;
	float maxx, maxy;
};

void bBox(vector<Point2D> points, BBox& bbox, size_t num)
{
	float minx = 1.0; float miny = 1.0;
	float maxx = 0.0; float maxy = 0.0;
	for (int i = 0; i < num; i++)
	{
		if (points[i].x < minx)minx = points[i].x;
		if (points[i].x > maxx)maxx = points[i].x;
		if (points[i].y < miny)miny = points[i].y;
		if (points[i].y > maxy)maxy = points[i].y;

	}
	bbox.minx = minx; bbox.maxx = maxx;
	bbox.miny = miny; bbox.maxy = maxy;
}

class HashValue
{
public:
	unsigned int morton;
	Point2D p;
	bool operator<(const HashValue& rhs) const {return  (morton < rhs.morton); }
	friend ostream& operator<<(ostream& os, const HashValue& h)
	{
		os << h.morton<<" "<<h.p;
		return os;
	}
};
__host__ __device__ void swap(HashValue& a, HashValue& b)
{
	HashValue tmp;
	tmp = a;
	a = b;
	b = tmp;
}



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


//attention, grid_dim is represented by 2 to the power of grid_dim
//grid_dim less than 2^16 (for long int)
__host__ __device__ unsigned int mortonHash2D(Point2D point, size_t grid_dim)
{
	int x_axis = 0;
	int y_axis = 0;
	x_axis = (int)((point.x / 1.0) * (1<<grid_dim));
	y_axis = (int)((point.y / 1.0) * (1<<grid_dim));

	int interleaved_x = 0;
	int interleaved_y = 0;
	int mark = 0x01;
	for (int i = 0; i < grid_dim; i++)
	{
		interleaved_x |= ((x_axis & mark) << i << i);
		x_axis = x_axis >> 1;

		interleaved_y |= ((y_axis & mark) << i << i);
		y_axis = y_axis >> 1;
	}

	return (interleaved_x << 1) | (interleaved_y);
}
__host__ __device__ unsigned int mortonHash2D_axis(int x, int y, size_t grid_dim)
{
	int interleaved_x = 0;
	int interleaved_y = 0;
	int mark = 0x01;

	for (int i = 0; i < grid_dim; i++)
	{
		interleaved_x |= ((x & mark) << i << i);
		x = x >> 1;

		interleaved_y |= ((y & mark) << i << i);
		y = y >> 1;
	}

	return (interleaved_x << 1) | (interleaved_y);
}
struct HashElem
{
	int idx;
	size_t num;
	friend ostream& operator<<(ostream& os, const HashElem& h)
	{
		os << h.idx<<" "<<h.num;
		return os;
	}
};

__host__ __device__ void minHeapify(HashValue* A, int i, int size)
{
	/*int l = left(i);
	int r = right(i);
	int smallest;
	if ((l <= (size - 1)) && (A[l].p.w < A[i].p.w))
	{
		smallest = l;
	}
	else
	{
		smallest = i;
	}
	if ((r <= (size - 1)) && (A[r].p.w < A[smallest].p.w))
	{
		smallest = r;
	}
	if (smallest != i)
	{
		swap(A[i], A[smallest]);
		minHeapify(A, smallest, size);
	}
*/
	int cur = i;
	int l; int r;
	while (cur < size)
	{
		l = left(cur);
		r = right(cur);
		int smallest;
		if ((l >= size) || (r >= size))break;
		else
		{
			if ((l < size) && (A[l].p.w < A[i].p.w))
			{
				smallest = l;
			}
			else
			{
				smallest = cur;
			}
			if ((r < size) && (A[r].p.w < A[smallest].p.w))
			{
				smallest = r;
			}
			if (smallest != cur)
			{
				swap(A[i], A[smallest]);
				cur = smallest;
			}
			else
			{
				break;
			}
		}
	}


}

__host__ __device__ HashValue minExtractHeap(HashValue*A,  int size)
{
	//if (size < 1) is put outside

	HashValue min = A[0];
	A[0] = A[size - 1];
	minHeapify(A, 1, size);
	return min;
}

__host__ __device__ void buildMinHeap(HashValue* A, int size)//All heaps are min heap
{
	for (int i = (size / 2 - 1); i >= 0; i--)
	{
		minHeapify(A, i, size);
	}
}




void MQ_heapify(vector<HashValue>& MQ_host, vector<HashElem>& MQ_idx_host, size_t point_num, size_t MQ_size)
{
	for (int iter_morton = 0; iter_morton < MQ_idx_host.size(); iter_morton++)
	{
		int sz = MQ_idx_host[iter_morton].num-1;//minus one is necessary if you want sz to be index.
		int idx = MQ_idx_host[iter_morton].idx;
		if (sz == 0)continue;
		else
		{
			buildMinHeap(&MQ_host[idx], sz);
		}
	}
}
//Hashing these points to a hashvalue vector

__global__ void dev_MQ_heapify(HashValue* MQ_dev, HashElem* MQ_idx_dev,size_t MQ_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < MQ_size)
	{
		int sz = MQ_idx_dev[idx].num;//minus one is necessary if you want sz to be index.
		int start_idx = MQ_idx_dev[idx].idx;
		if (sz == 0)return;
		else
		{
			buildMinHeap(&MQ_dev[start_idx], sz);
		}
	}
}

void hashingPoints2D(vector<Point2D>& points, vector<HashValue>& hash, size_t size, size_t grid_dimension)
{

	for (int i = 0; i < size; i++)
	{
		hash[i].morton = mortonHash2D(points[i], grid_dimension);
		hash[i].p = points[i];
	}

	//sort hash array
	sort(hash.begin(), hash.end());

	//the sorted hash array is the raw data of HashGrid.
	//the hash grid  need another array for indexing: HashIdx.

}
void hashingPoints2DWeighted(vector<HashValue>& points, vector<HashValue>& hash, size_t size, size_t grid_dimension)
{
	hash.resize(size);
	for (int i = 0; i < size; i++)
	{
		hash[i].morton = mortonHash2D(points[i].p, grid_dimension);
		hash[i].p = points[i].p;
	}
	sort(hash.begin(), hash.end());
}


//this is going to generate a marker array
__host__ void generateMarkerArray(vector<int>& marker, vector<HashValue>& hash_grids, size_t size)
{
	for (int i = 0; i < size; i++)
	{
		if (i == 0)marker[i] = 1;
		else
		{
			if (hash_grids[i].morton != hash_grids[i - 1].morton)
			{
				marker[i] = 1;
			}
			else
			{
				marker[i] = 0;
			}
		}
	}
}





void hashIndexing(vector<HashValue>& hash_grids, vector<HashElem>& hash_idx, size_t size)
{


	//Prefix sum to get each morton code count;
	vector<int> marker;
	marker.resize(size);
	generateMarkerArray(marker, hash_grids, size);

	/*thrust::device_vector<int> offset_dev = marker;
	thrust::inclusive_scan(offset_dev.begin(), offset_dev.end(), offset_dev.begin());
	thrust::host_vector<int> offset_host(offset_dev.begin(), offset_dev.end());
*/
	//Now we have offset_host, we can do the indexing.

	//To make it easier, we make a count array.
	//each marker's index minus privious index(where marker is 1).
	vector<int> count;
	count.resize(size);
	int current_marker_offset = 0;
	int ct = 1;//counting
	for (int i = 0; i < size; i++)//iterate marker
	{
		if (marker[i] == 1)
		{
			count[current_marker_offset] = ct;
			ct = 1;
			current_marker_offset = i;//start counting
		}
		else
			ct++;
	}
	count[current_marker_offset] = ct;//solve the last = 0 bug

	//fill the hash_idx array
	for (int i = 0; i < size; i++)
	{
		if (marker[i] == 1)
		{
			int morton = hash_grids[i].morton;
			hash_idx[morton].idx = i;
			hash_idx[morton].num = count[i];
		}
		else
		{
			continue;
		}
	}
}


void matlabView(vector<Point2D> points, string filename)
{
	const string dir = "C:/Users/lunamu/Dropbox/MATLAB/";		//it's called matlabView for a reason
	ofstream file(dir + filename);
	for (int i = 0; i < points.size(); i++)
	{
		file << points[i].x << " " << points[i].y << endl;
	}
}

void axis(Point2D p, size_t grid_dim)
{
	int x_axis = (int)((p.x / 1.0) * (1<<grid_dim));
	int y_axis = (int)((p.y / 1.0) * (1<<grid_dim));
}

__host__ __device__ float distance2(Point2D p1, Point2D p2)
{
	return sqrtf((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

#define EPS 0.00001
__global__ void devAssignOriginalWeighting(HashValue* hash_grids, HashElem* hash_idx, float radius, size_t grid_dim, size_t point_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < point_num)
	{
		Point2D& current_point = hash_grids[idx].p;
		Point2D& min_corner = (current_point - radius);
		Point2D& max_corner = (current_point + radius);
		int x_axis_start = (int)((min_corner.x / 1.0) * (1 << grid_dim));
		int y_axis_start = (int)((min_corner.y / 1.0) * (1 << grid_dim));
		int x_axis_end = (int)((max_corner.x / 1.0) * (1 << grid_dim));
		int y_axis_end = (int)((max_corner.y / 1.0) * (1 << grid_dim));

		for (int x = x_axis_start; x <= x_axis_end; x++)
		{
			for (int y = y_axis_start; y <= y_axis_end; y++)
			{
				int morton = mortonHash2D_axis(x, y, grid_dim);
				int offset = hash_idx[morton].idx;
				for (int idx_in_grid = 0; idx_in_grid < hash_idx[morton].num; idx_in_grid++)
				{
					float d = distance2(hash_grids[offset + idx_in_grid].p, current_point);
					//if (d < radius)current_point.w += d;
					//test, only use the counting
					if (d < EPS) continue;
					else if (d < radius)current_point.w += 1.0/d;
				}
			}
		}
	}
}
//only for test!
__global__ void OriginalIndexDevAssignOriginalWeighting(Point2D* points, HashValue* hash_grids, HashElem* hash_idx, float radius, size_t grid_dim, size_t point_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < point_num)
	{
		Point2D& current_point = points[idx];
		Point2D& min_corner = (current_point - radius);
		Point2D& max_corner = (current_point + radius);
		int x_axis_start = (int)((min_corner.x / 1.0) * (1 << grid_dim));
		int y_axis_start = (int)((min_corner.y / 1.0) * (1 << grid_dim));
		int x_axis_end = (int)((max_corner.x / 1.0) * (1 << grid_dim));
		int y_axis_end = (int)((max_corner.y / 1.0) * (1 << grid_dim));

		for (int x = x_axis_start; x <= x_axis_end; x++)
		{
			for (int y = y_axis_start; y <= y_axis_end; y++)
			{
				int morton = mortonHash2D_axis(x, y, grid_dim);
				int offset = hash_idx[morton].idx;
				for (int idx_in_grid = 0; idx_in_grid < hash_idx[morton].num; idx_in_grid++)
				{
					float d = distance2(hash_grids[offset + idx_in_grid].p, current_point);
					//if (d < radius)current_point.w += d;
					//test, only use the counting
					if (d < radius)current_point.w += 1;
				}
			}
		}
	}
}
__global__ void extractBatch(HashValue* dev_hash_grids, HashElem* dev_hash_idx, size_t morton_num, HashValue* results, size_t batchSize, int* randomized_grid_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < batchSize)
	{
		int current_idx = randomized_grid_idx[idx];//map current randomized idx;
		int heap_idx = dev_hash_idx[current_idx].idx;
		int heap_size = dev_hash_idx[current_idx].num;
		if (heap_size < 1)
		{
			HashValue nil; nil.morton = 0; nil.p.x = 0; nil.p.y = 0; nil.p.w = 0;
			results[current_idx] = nil;
			return;
		}
		results[idx] = minExtractHeap(&dev_hash_grids[heap_idx], heap_size);

		//mega kernel, update in one kernel vs. update in different kernel;
		//currently update in different kernel;

		dev_hash_idx[current_idx].num -= 1;
	}
}


//To understand
//if radius is smaller than the radius recorded in w, then replace radius.
//but the heap is min heap, and we want to select w with large r, so radius is recorded in reciprocal form;
__host__ __device__ float weighting(Point2D target_point, Point2D query_point, float r)
{
	if (r == 0) return 0.0;//query point it self;
	else if (target_point.w - 0.0 < EPS) return 1.0 / r;
	else if (r < 1. / target_point.w) return 1.0 / r;
	else return target_point.w;
}
__global__ void updateWeight(HashValue* dev_hash_grids, HashElem* dev_hash_idx, HashValue* batch, float radius, size_t grid_dim, size_t batchSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < batchSize)
	{
		Point2D query_point = batch[idx].p;
		Point2D& min_corner = (query_point - radius);
		Point2D& max_corner = (query_point + radius);
		int x_axis_start = (int)((min_corner.x / 1.0) * (1 << grid_dim));
		int y_axis_start = (int)((min_corner.y / 1.0) * (1 << grid_dim));
		int x_axis_end = (int)((max_corner.x / 1.0) * (1 << grid_dim));
		int y_axis_end = (int)((max_corner.y / 1.0) * (1 << grid_dim));
		for (int x = x_axis_start; x <= x_axis_end; x++)
		{
			for (int y = y_axis_start; y <= y_axis_end; y++)
			{
				int morton = mortonHash2D_axis(x, y, grid_dim);
				int offset = dev_hash_idx[morton].idx;
				for (int idx_in_grid = 0; idx_in_grid < dev_hash_idx[morton].num; idx_in_grid++)
				{
					float d = distance2(dev_hash_grids[offset + idx_in_grid].p, query_point);
					//if (d < radius)current_point.w += d;
					//test, only use the counting
					if (d < EPS) continue;
					else if (d < radius)
					{
						//weighting strategy!
						dev_hash_grids[offset + idx_in_grid].p.w = weighting(dev_hash_grids[offset + idx_in_grid].p, query_point, d);

						//TODO::profile, immediately triger a decrease key (atomicly), or update the whole afterwards?
					}
				}
			}
		}

	}
}


void heapifytest(HashValue* MQ_dev, HashElem* MQ_idx_dev, size_t MQ_size)
{
	int sz = 49;//minus one is necessary if you want sz to be index.
	int idx = 0;
	if (sz == 0)return;
	else
	{
		buildMinHeap(&MQ_dev[idx], sz);
	}
}

__global__ void resetWeighting(HashValue* dev_hash_grids, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		dev_hash_grids[idx].p.w = 0.0;
	}
}

__global__ void generateRandomPoint(curandState_t* states, Point2D* points, unsigned int seed, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		curand_init(seed * idx, blockIdx.x, 0, &states[idx]);
		points[idx].x =  (curand(&states[idx]) % RAND_MAX) / float(RAND_MAX);
		points[idx].y =  (curand(&states[idx]) % RAND_MAX) / float(RAND_MAX);
		points[idx].w = 0.0;
	}
}


__global__ void hashingPoints2Ddev(Point2D* dev_points, HashValue* hash, size_t size, size_t grid_dimension)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		hash[idx].morton = mortonHash2D(dev_points[idx], grid_dimension);
		hash[idx].p = dev_points[idx];
	}
}
//
//__global__ void init_curand(curandState_t* states, unsigned int seed, size_t size)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < size)
//	{
//		curand_init(seed, blockIdx.x, 0, &states[idx]);
//	}

struct marker_struct
{
	int marker;
	int morton;
};
__global__ void generateMarkerArray(marker_struct* marker, HashValue* hash_grids, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		if (idx == 0)
		{
			marker[idx].marker = 1;
			marker[idx].morton = hash_grids[idx].morton;
		}
		else
		{
			if (hash_grids[idx].morton != hash_grids[idx - 1].morton)
			{
				marker[idx].marker = 1;
				marker[idx].morton = hash_grids[idx].morton;
			}
			else
			{
				marker[idx].marker = 0;
				marker[idx].morton = 0;
			}
		}
	}
}
__global__ void generateCounterArray(marker_struct* marker, HashElem* counter, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		counter[idx].idx = idx;
		int ct = 1;
		if (marker[idx].marker == 1)
		{
			int cur = idx+1;
			while ((cur < size) && (marker[cur].marker == 0))
			{
				ct++;
				cur++;
			}
			counter[idx].num = ct;
		}
		else
			counter[idx].num = 0;
	}
}

__global__ void  hashIndexing(HashValue* hash_grids, HashElem* hash_idx, int* marker, size_t size)
{


}


//struct for thrust execution
struct HashCmp {
	__host__ __device__
		bool operator()(const HashValue& o1, const HashValue& o2) {
		return o1.morton < o2.morton;
	}
};

struct marker_not_zero
{
	__host__ __device__
	bool operator()(marker_struct x)
	{
		return x.marker != 0;
	}
};


int main(int argc, char *argv[])
{
	cudaFree(0);
	//important parameters
	int grid_dim = 7;
	int point_num = 1000000;
	// float radius = 0.001;


	// int grid_dim = atoi(argv[1]);										//Grid_dimension, 1<<grid_dim is total number (1D)
	// int point_num = atoi(argv[2]);								//Point Number
	float radius = 0.5337 / sqrtf(point_num);				//rmax on a 2d surface
	int morton_num = (1 << grid_dim) * (1 << grid_dim);		//morton num of 2D

	vector<Point2D> points(point_num);									//host points 2D raw
	vector<HashValue> hash_grids(point_num);							//actual storage of all points
	vector<HashElem> hash_idx(morton_num);								//index & num of a certain grid, size is morton num.
	generateRandomPointCloud(points, point_num);
	/* Fills the public  */

	hashingPoints2D(points, hash_grids, point_num, grid_dim);
	hashIndexing(hash_grids, hash_idx, point_num);

	//Now, all important index are built, move to GPU;

	//copy to GPU
	HashValue* dev_hash_grids = 0;
	HashElem* dev_hash_idx = 0;
	GPUCHECK(cudaMalloc((void**)&dev_hash_grids, sizeof(HashValue)*point_num));
	GPUCHECK(cudaMalloc((void**)&dev_hash_idx, sizeof(HashElem)*morton_num));
	GPUCHECK(cudaMemcpy(dev_hash_grids, &hash_grids[0], point_num * sizeof(HashValue), cudaMemcpyHostToDevice));
	GPUCHECK(cudaMemcpy(dev_hash_idx, &hash_idx[0], morton_num * sizeof(HashElem), cudaMemcpyHostToDevice));



	//generate original weighting
	//1. assign each weighting as num of points within radius.
	//host version
	//assignOriginalWeighting(hash_grids, hash_idx, radius, grid_dim);

	int threadsPerBlock = 256;
	int numBlocks = (point_num + threadsPerBlock - 1) / threadsPerBlock;

	//This is only going to compare with the results from MATLAB's radius search.
	//OriginalIndexDevAssignOriginalWeighting << <numBlocks, threadsPerBlock >> >(dev_points, dev_hash_grids, dev_hash_idx, radius, grid_dim, point_num);

	//dev weighting assignment.
	devAssignOriginalWeighting << <numBlocks, threadsPerBlock >> >(dev_hash_grids, dev_hash_idx, radius, grid_dim, point_num);
	//viewGPUArray<HashValue>(dev_hash_grids, point_num, "bc_dev_hash_grids");
	//viewGPUArray<HashValue>(dev_hash_grids, point_num, "dev_hash_grids");
	//heapify
	numBlocks = (morton_num + threadsPerBlock - 1) / threadsPerBlock;
	dev_MQ_heapify << <numBlocks, threadsPerBlock >> >(dev_hash_grids, dev_hash_idx,  morton_num);

	//HashValue* host_hash_grids = (HashValue*)malloc(sizeof(HashValue) * point_num);
	//HashElem* host_hash_idx = (HashElem*)malloc(sizeof(HashValue) * point_num);
	//vector<HashValue> host_hash_grids(point_num);
	//vector<HashElem> host_hash_idx(morton_num);
	//GPUCHECK(cudaMemcpy(&host_hash_grids[0], dev_hash_grids, sizeof(HashValue) * point_num, cudaMemcpyDeviceToHost));
	//GPUCHECK(cudaMemcpy(&host_hash_idx[0], dev_hash_idx, sizeof(HashElem) * morton_num, cudaMemcpyDeviceToHost));


	//heapifytest(&host_hash_grids[0], &host_hash_idx[0], morton_num);

	//viewGPUArray<HashValue>(dev_hash_grids, point_num, "dev_hash_grids");
	//viewGPUArray<HashElem>(dev_hash_idx, morton_num, "dev_hash_idx");
	vector<int> randomized_index_array;

	int batchSize = morton_num / 4;
	int desiredNum = batchSize * 16;

	int* dev_randomized_index_array;
	randomized_index_array.resize(batchSize);
	for (int i = 0; i < randomized_index_array.size(); i++)
	{
		//TODO:randomize
		randomized_index_array[i] = i;
	}

	HashValue* desiredResults;
	GPUCHECK(cudaMalloc((void**)&dev_randomized_index_array, sizeof(int) * batchSize));
	GPUCHECK(cudaMalloc((void**)&desiredResults, sizeof(HashValue) * desiredNum));
	GPUCHECK(cudaMemcpy(dev_randomized_index_array, &randomized_index_array[0], sizeof(int) * batchSize, cudaMemcpyHostToDevice))

	int threadsPerBlock_batch = 64;
	int numBlocks_batch = (batchSize + threadsPerBlock_batch - 1) / threadsPerBlock_batch;
	//extract those calculated with radius search


	//viewGPUArray<HashElem>(dev_hash_idx, morton_num, "bc_dev_hash_idx");

	extractBatch << <numBlocks_batch, threadsPerBlock_batch >> >(dev_hash_grids, dev_hash_idx, morton_num, desiredResults, batchSize, dev_randomized_index_array);
	GPUCHECK(cudaPeekAtLastError());
	GPUCHECK(cudaDeviceSynchronize());

	viewGPUArray<HashValue>(dev_hash_grids, point_num, "dev_hash_grids.txty");
	//reset weighting
	int threadsPerBlock_reset_weighting = 256;
	int numBlocks_reset_weighting = (point_num + threadsPerBlock_reset_weighting - 1) / threadsPerBlock_reset_weighting;
	resetWeighting << <numBlocks_reset_weighting, threadsPerBlock_reset_weighting >> >(dev_hash_grids, point_num);
	GPUCHECK(cudaPeekAtLastError());
	GPUCHECK(cudaDeviceSynchronize());

	//update weighting
	int threadsPerBlock_updateWeighting = 64;
	int numBlocks_updateWeighting = (batchSize + threadsPerBlock_updateWeighting - 1) / threadsPerBlock_updateWeighting;
	int offset = 0;//offset of this batch
	updateWeight << <numBlocks_updateWeighting, threadsPerBlock_updateWeighting >> >(dev_hash_grids, dev_hash_idx, desiredResults, radius, grid_dim, batchSize);
	GPUCHECK(cudaPeekAtLastError());
	GPUCHECK(cudaDeviceSynchronize());


	for (int batch_idx = 1; batch_idx < desiredNum/batchSize; batch_idx++)
	{
		int tpb_heapify = 256;
		int nb_heapify = (point_num + tpb_heapify - 1) / tpb_heapify;
		dev_MQ_heapify << <nb_heapify, tpb_heapify >> >(dev_hash_grids, dev_hash_idx, morton_num);
		GPUCHECK(cudaPeekAtLastError());
		GPUCHECK(cudaDeviceSynchronize());

		int tpb_extract = 64;
		int nb_extract = (batchSize + tpb_extract - 1) / tpb_extract;
		extractBatch << <nb_extract, tpb_extract >> >(dev_hash_grids, dev_hash_idx, morton_num, desiredResults + batchSize*batch_idx, batchSize, dev_randomized_index_array);
		GPUCHECK(cudaPeekAtLastError());
		GPUCHECK(cudaDeviceSynchronize());

		int tpb_update = 64;
		int nb_update = (batchSize + tpb_update - 1) / threadsPerBlock_updateWeighting;
		updateWeight << <nb_update, tpb_update >> >(dev_hash_grids, dev_hash_idx, desiredResults + batch_idx * batchSize, radius, grid_dim, batchSize);
		GPUCHECK(cudaPeekAtLastError());
		GPUCHECK(cudaDeviceSynchronize());
	}

	//viewGPUArray<HashValue>(desiredResults, desiredNum, "batch");
	//viewGPUArrayMat<HashValue>(desiredResults, desiredNum, "batch");

	GPUCHECK(cudaFree(dev_hash_grids));
	GPUCHECK(cudaFree(dev_hash_idx));
	//GPUCHECK(cudaFree(dev_points));
	GPUCHECK(cudaFree(desiredResults));
	GPUCHECK(cudaFree(dev_randomized_index_array));

    return 0;
}

//
////time test
//#ifdef TEST
//for (int i = 10000; i <= 500000; i += 10000)
//{
//	generateRandomPointCloud(points, i);
//	int start_s = clock();
//	hashingPoints2D(points, hash_grids, i, grid_dim);
//	//make hash index (used for searching)
//
//	hashIndexing(hash_grids, hash_idx, i);
//	int stop_s = clock();
//
//	cout << i << " " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000.0 << endl;
//}
//#endif



//All efforts to build on device.

//Point2D* dev_points;
//curandState_t* states;
//GPUCHECK(cudaMalloc((void**)&dev_points, sizeof(Point2D) * point_num));
//GPUCHECK(cudaMalloc((void**)&states, sizeof(curandState_t) * point_num));
//
//int tpb_generate = 256;
//int nb_generate = (point_num + tpb_generate - 1) / tpb_generate;
//generateRandomPoint << <nb_generate, tpb_generate >> >(states, dev_points, time(0), point_num);

////dev generated random too slow, use host.
////generateRandomPointCloud(points, point_num);
////GPUCHECK(cudaMemcpy(dev_points, &points[0], sizeof(Point2D) * point_num, cudaMemcpyHostToDevice));


//viewGPUArray<Point2D>(dev_points, point_num, "points");
////GPUCHECK(cudaMemcpy(&points[0], dev_points, sizeof(Point2D) * point_num, cudaMemcpyDeviceToHost));


/**************************************************************************************/
//Step 2. Generate Point Hashing (Morton)
/**************************************************************************************/
//HashValue* dev_hash_grids = 0;
//HashElem* dev_hash_idx = 0;

//GPUCHECK(cudaMalloc((void**)&dev_hash_grids, point_num * sizeof(HashValue)));
//GPUCHECK(cudaMalloc((void**)&dev_hash_idx, morton_num * sizeof(HashElem)));
//
//int tpb_hashing = 256;
//int nb_hashing = (point_num + tpb_hashing - 1) / tpb_hashing;
//hashingPoints2Ddev << <nb_hashing, tpb_hashing >> >(dev_points, dev_hash_grids, point_num, grid_dim);
//GPUCHECK(cudaPeekAtLastError());
//GPUCHECK(cudaDeviceSynchronize());
//marker_struct* marker;
//HashElem* counter;
//GPUCHECK(cudaMalloc((void**)&marker, sizeof(marker_struct) * point_num));
//GPUCHECK(cudaMalloc((void**)&counter, sizeof(HashElem) * point_num));

//thrust::device_ptr<HashValue> dev_hash_grids_pointer(dev_hash_grids);
//thrust::sort(dev_hash_grids_pointer, dev_hash_grids_pointer + point_num, HashCmp());
//
//
//int tpb_marking = 256;
//int nb_marking = (point_num + tpb_marking - 1) / tpb_marking;
//generateMarkerArray << <nb_marking, tpb_marking >> >(marker, dev_hash_grids, point_num);
//GPUCHECK(cudaPeekAtLastError());
//GPUCHECK(cudaDeviceSynchronize());
////viewGPUArray<marker_struct>(marker, point_num, "marker");

//
//generateCounterArray << <nb_marking, tpb_marking >> >(marker, counter, point_num);
//GPUCHECK(cudaPeekAtLastError());
//GPUCHECK(cudaDeviceSynchronize());
//viewGPUArray<HashElem>(counter, point_num, "counter");
//thrust::device_ptr<marker_struct> marker_pointer(marker);
//thrust::device_ptr<HashElem> counter_pointer(counter);

//thrust::copy_if(thrust::device,counter_pointer, counter_pointer + point_num, marker_pointer, dev_hash_idx, marker_not_zero());
//viewGPUArray<HashElem>(dev_hash_idx, morton_num, "dev_hash_idx");



//hashIndexing(hash_grids, hash_idx, point_num);
