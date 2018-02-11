#include <cstdlib>
#include <vector>
#include <iostream>
#include <numeric>
#include "CPUTimer.h"
#include "GPUTimer.h"
#include "CUDAError.h"


void GenerateRandomArray(int* array, const int N)
{
	srand(time(NULL));
	for (int i = 0; i < N; ++i) {
		float tmp = rand() / (float)RAND_MAX * 100.0f;
		array[i] = tmp - 50;
	}
}

int AddSequentialInCPU(const int* array, const int N)
{
	int sum = 0;
	for(int i = 0; i < N; ++i)
		sum += array[i];
	return sum;
}

__device__ void FirstAddWithCopyOnShared(const int* in, int* shared, const int index, const unsigned int stepSize)
{
	if(threadIdx.x < stepSize)
		shared[threadIdx.x] = in[index] + in[index + stepSize];
	__syncthreads();
}

__device__ inline int isFirstThreadOfThisBlock()
{
	return threadIdx.x == 0;
}

__device__ void Reduce(int* in, int* out, const int index, unsigned int stepSize)
{
	while(stepSize > 0)
	{
		if(threadIdx.x < stepSize)
			in[index] += in[index + stepSize];
		else
			break;
		stepSize >>= 1;
		__syncthreads();
	}

	if(isFirstThreadOfThisBlock())
		out[blockIdx.x] = in[index];
}

__global__ void AddParallelInGPU(int* in, int* out)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ int s_data[];
	FirstAddWithCopyOnShared(in, s_data, index, blockDim.x >> 1);
	Reduce(s_data, out, threadIdx.x, blockDim.x >> 2);
}

int main()
{
	const int N = 1 << 20;
	int* h_array = (int*)malloc(N * sizeof(int));
	int cpuSum = 0;
	GenerateRandomArray(h_array, N);


	int h_gpuSum[1];
	int* d_array;
	int* d_sumBlocks;
	int *d_gpuSum;
	const int BLOCK_DIM = 1024;
	const int GRID_DIM = N / BLOCK_DIM;
	cudaMalloc((void**)&d_array, N * sizeof(int));
	cudaMalloc((void**)&d_sumBlocks, GRID_DIM * sizeof(int));
	cudaMalloc((void**)&d_gpuSum, sizeof(int));

	cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

	AddParallelInGPU<<<GRID_DIM, BLOCK_DIM, (BLOCK_DIM >> 1) * sizeof(int)>>>(d_array, d_sumBlocks);
	AddParallelInGPU<<<1, GRID_DIM, (GRID_DIM >> 1) * sizeof(int)>>>(d_sumBlocks, d_gpuSum);

	cpuSum = AddSequentialInCPU(h_array, N);

	cudaDeviceSynchronize();
	cudaMemcpy(h_gpuSum, d_gpuSum, sizeof(int), cudaMemcpyDeviceToHost);

	if(cpuSum == h_gpuSum[0])
		std::cout << "MATCH";
	else
		std::cout << "WRONG";
	std::cout << std::endl;

	cudaFree(d_array);
	cudaFree(d_sumBlocks);
	cudaFree(d_gpuSum);

	return 0;
}
