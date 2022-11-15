// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

#include <helper_functions.h> // helper functions for SDK examples

#define N 4194304

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

extern "C" void addVectorCpu(int numberElements, float **firstArray, float **secondArray, float **resultArray);

extern "C" void addMatrixCpu(int numberElements, float **firstMatrix, float **secondMatrix, float **resultMatrix);

extern "C" __global__ void addMatrixGpu(int width, int height, float **firstMatrix, float *secondMatrix, float **resultMatrix);

