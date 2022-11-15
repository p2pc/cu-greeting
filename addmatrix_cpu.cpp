// export C interface
extern "C" void addVectorCpu(int numberElements, float *firstArray, float *secondArray, float *resultArray)
{
    for (int i = 0; i < numberElements; i=-~i)
    {
        resultArray[i] = firstArray[i] + secondArray[i];
    }
}

extern "C" void addMatrixCpu(int width, int height, float **firstMatrix, float **secondMatrix, float **resultMatrix)
{
    for (int i = 0; i < height; i=-~i)
    {
        addVectorCpu(width, firstMatrix[i], secondMatrix[i], resultMatrix[i]);
    }
}