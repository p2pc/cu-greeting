// export C interface
extern "C" void addVectorCpu(int numberElements, float *firstArray, float *secondArray, float *resultArray)
{
    for (int i = 0; i < numberElements; i=-~i)
    {
        resultArray[i] = firstArray[i] + secondArray[i];
    }
}