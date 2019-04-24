/*WBL 21 March 2009 $Revision: 1.14 $
 * based on cuda/sdk/projects/quasirandomGenerator/quasirandomGenerator_kernel.cuh
 */



#ifndef PARKMILLER_KERNEL_CUH
#define PARKMILLER_KERNEL_CUH


#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include "realtype.h"
#include "park-miller_common.h"



//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)



////////////////////////////////////////////////////////////////////////////////
// Park-Miller quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////

static __global__ void parkmillerKernel(
    int *d_Output,
    unsigned int seed,
    int cycles,
    unsigned int N
){
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);
double const a    = 16807;      //ie 7**5
double const m    = 2147483647; //ie 2**31-1
double const reciprocal_m = 1.0/m;

    for(unsigned int pos = tid; pos < N; pos += threadN){
        unsigned int result = 0;
        unsigned int data = seed + pos;

	for (int i=1; i<=cycles; i++) {

// W. Langdon cs.ucl.ac.uk 5 May 1994

//inline int intrnd (int& seed) // 1<=seed<=m
{
#ifndef DOUBLE_PRECISION
eror dont compile INT64 version slower than double precision version on GT200
even with 64 bit Linux under CUDA 3.1
//#ifdef LONG_GE46BITS
//int const a    = 16807;      //ie 7**5
//int const m    = 2147483647; //ie 2**31-1
	data = d_Output[pos] = (INT64(data) * 16807)%2147483647;
	//data = result;
	//return seed;
#else
	double temp = data * a;
	result = (int) (temp - m * floor ( temp * reciprocal_m ));
	data = result;
	//cout<<"seed "<<seed<<endl;
	//return seed;
#endif
}//end mycode
}//endfor

        //d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
        //d_Output[MUL(threadIdx.y, N) + pos] = result;
        d_Output[pos] = result;
    }
}

//Host-side interface
static void parkmillerGPU(int *d_Output, unsigned int seed, int cycles,
       unsigned int grid_size, unsigned int block_size, unsigned int N){
    parkmillerKernel<<<grid_size, block_size>>>(d_Output, seed, cycles, N);
    cutilCheckMsg("parkmillerKernel() execution failed.\n");
}

#endif
