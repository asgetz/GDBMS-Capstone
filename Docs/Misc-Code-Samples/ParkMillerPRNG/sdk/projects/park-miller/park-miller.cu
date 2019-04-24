/*WBL 21 March 2009 $Revision: 1.3 $
 * based on cuda/sdk/projects/quasirandomGenerator/quasirandomGenerator_SM13.cu
 */



#define DOUBLE_PRECISION
#include "park-miller_kernel.cuh"


extern "C" void parkmiller_gpu(int *d_Output, unsigned int seed, int cycles, 
unsigned int grid_size, unsigned int block_size, unsigned int N){
    parkmillerGPU(d_Output, seed, cycles, grid_size, block_size, N);
}


