// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  printf("[LOG] Thread ID: %d\n", tid);

  x[tid] = (float)threadIdx.x;
}

int main(int argc, const char **argv)
{
  float *h_x, *d_x, *k_x, *e_x;

  int nblocks, nthreads, nsize, n;

  // initialise card
  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block
  nblocks = 2;
  nthreads = 8;
  nsize = nblocks * nthreads;

  // allocate memory for array
  h_x = (float *)malloc(nsize * sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize * sizeof(float)));

  k_x = (float *)malloc(nsize * sizeof(float));
  checkCudaErrors(cudaMalloc((void **)&e_x, nsize * sizeof(float)));

  // execute kernel
  my_first_kernel<<<nblocks, nthreads>>>(d_x);
  getLastCudaError("my_first_kernel execution failed\n");
  my_first_kernel<<<nblocks, nthreads>>>(e_x);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out
  checkCudaErrors(cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(k_x, e_x, nsize * sizeof(float), cudaMemcpyDeviceToHost));

  for (n = 0; n < nsize; n++)
  {
    printf(" n,  h_x, k_x, sum(h_x, k_x)  =  %d  %f %f %f\n", n, h_x[n], k_x[n], h_x[n] + k_x[n]);
  }

  // free memory
  checkCudaErrors(cudaFree(d_x));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

  return 0;
}