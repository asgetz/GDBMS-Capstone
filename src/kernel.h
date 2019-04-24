#ifndef KERNEL_H
#define KERNEL_H

void bootstrapGPU(unsigned int blocknum, unsigned int threadnum,
                  unsigned int *output, int *sampleInput, curandState *state);

void addKernelGPU(unsigned int blocknum, unsigned int threadnum,
                  int *c, const int *a, const int *b);

void parkmillerGPU(int *d_Output, unsigned int seed, int cycles,
                          unsigned int grid_size,
                          unsigned int block_size,
                          unsigned int N);

#endif
