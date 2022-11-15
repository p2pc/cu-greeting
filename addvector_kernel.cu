
extern "C" __global__ void addVectorGpu(int numberElements, float *firstArray, float *secondArray, float *resultArray)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numberElements)
    {
        resultArray[i] = firstArray[i] + secondArray[i];
    }
}