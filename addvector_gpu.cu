// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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

// includes CUDA
#include <cuda_runtime.h>

#include <helper_functions.h> // helper functions for SDK examples

extern "C" void addVectorCpu(int numberElements, float *firstArray, float *secondArray, float *resultArray);

extern "C" __global__ void addVectorGpu(int numberElements, float *firstArray, float *secondArray, float *resultArray);

void addVec(int numberElements, float *firstArray, float *secondArray, float *resultArray, bool useDevice = false)
{
    StopWatchInterface *timer = 0;

    if (useDevice == false)
    {
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        addVectorCpu(numberElements, firstArray, secondArray, resultArray);

        sdkStopTimer(&timer);
        printf("Processing time on CPU: %f (ms)\n", sdkGetTimerValue(&timer));
    }
    else
    {
        cudaDeviceProp devProv;
        cudaGetDeviceProperties(&devProv, 0);
        printf("**********GPU info**********\n");
        printf("Name: %s\n", devProv.name);
        printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
        printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
        printf("****************************\n");

        // Host allocates memories on device
        // Way 1:
        float *d_firstArray, *d_secondArray, *d_resultArray;
        size_t nBytes = numberElements * sizeof(float);

        CHECK(cudaMalloc(&d_firstArray, nBytes));
        CHECK(cudaMalloc(&d_secondArray, nBytes));
        CHECK(cudaMalloc(&d_resultArray, nBytes));

        // Host copies data to device memories
        CHECK(cudaMemcpy(d_firstArray, firstArray, nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_secondArray, secondArray, nBytes, cudaMemcpyHostToDevice));

        // Host invokes kernel function to add vectors on device
        dim3 blockSize(512);                                   // For simplicity, you can temporarily view blockSize as a number
        dim3 gridSize((numberElements - 1) / blockSize.x + 1); // Similarity, view gridSize as a number

        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        addVectorGpu<<<gridSize, blockSize>>>(numberElements, d_firstArray, d_secondArray, d_resultArray);

        cudaDeviceSynchronize();

        sdkStopTimer(&timer);
        printf("Processing time on GPU: %f (ms)\n", sdkGetTimerValue(&timer));

        // Host copies result from device memory
        CHECK(cudaMemcpy(resultArray, d_resultArray, nBytes, cudaMemcpyDeviceToHost));

        // Free device memories
        CHECK(cudaFree(d_firstArray));
        CHECK(cudaFree(d_secondArray));
        CHECK(cudaFree(d_resultArray));
    }

    sdkDeleteTimer(&timer);
}

int main(int argc, char **argv)
{
    float *firstArray, *secondArray;         // Input vectors
    float *resultArray, *correctResultArray; // Output vector

    // Allocate memories for firstArray, secondArray, resultArray, correctResultArray
    size_t nBytes = N * sizeof(float);

    // Allocate the host input vector A (the first vector)
    firstArray = reinterpret_cast<float *>(malloc(nBytes));

    // Allocate the host input vector B (the second vector)
    secondArray = reinterpret_cast<float *>(malloc(nBytes));

    // Allocate the host input vector C (the result vector)
    resultArray = reinterpret_cast<float *>(malloc(nBytes));

    // Allocate the host input vector D (the correct vector which computed on host)
    correctResultArray = reinterpret_cast<float *>(malloc(nBytes));

    // Input data into in1, in2
    for (int i = 0; i < N; i++)
    {
        firstArray[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        secondArray[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    // Add vectors (on host)
    addVec(N, firstArray, secondArray, correctResultArray);

    // Add in1 & in2 on device
    addVec(N, firstArray, secondArray, resultArray, true);

    // Check correctness
    for (int i = 0; i < N; i = -~i)
    {
        if (resultArray[i] != correctResultArray[i])
        {
            printf("INCORRECT.\n");
            return 1;
        }
    }
    printf("CORRECT.\n");
}
