// This example is taken from a power point lecture covering the topic

///     HOST FUNCTIONS      ///
///     DEVICE FUNCTIONS    ///
///     KERNELS             ///
__global__ void reduce0(int *g_idata, int *g_odata){
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid]=g_idata[i];
    __syncthreads();

    // do reduction in shared memory
    for(unsigned int s=1;s<blockDim.x;s*=2){
        if(tid % (2*s) == 0){
            sdata[tid]+=sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if(tid==0)g_odata[blockIdx.x]=sdata[0];
}

__global__ void reduce1(int *g_idata, int *g_odata){
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    // Perform first level of reduction here
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // do reduction in shared memory
    // for(unsigned int s=1;s<blockDim.x;s*=2){
    //     int index = 2*s*tid;
    //     if(index<blockDim.x){
    //         sdata[tid]+=sdata[tid + s];
    //     }
    //     __syncthreads();
    // }
    for(unsigned int s=blockDim.x/2;s>32;s>>=1){
        if(tid<s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }

    // // write result for this block to global mem
    // if(tid == 0) g_odata[blockIdx.x] = sdata[0];

    if(tid<32){
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid +  8];
        sdata[tid] += sdata[tid +  4];
        sdata[tid] += sdata[tid +  2];
        sdata[tid] += sdata[tid +  1];
    }
}


template<unsigned int blockSize>
__global__ void reduce5(int *g_idata, int *g_odata){
    extern __shared__ int sdata[];

    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid <  64)  { sdata[tid] += sdata[tid +   64]; } __syncthreads();
    }

    if (tid < 32){
        if (blockSize >=  64)sdata[tid] += sdata[tid + 32];
        if (blockSize >=  32)sdata[tid] += sdata[tid + 16];
        if (blockSize >=  16)sdata[tid] += sdata[tid +  8];
        if (blockSize >=   8)sdata[tid] += sdata[tid +  4];
        if (blockSize >=   4)sdata[tid] += sdata[tid +  2];
        if (blockSize >=   2)sdata[tid] += sdata[tid +  1];
    }
}
