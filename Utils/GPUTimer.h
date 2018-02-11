#pragma once
#include <cuda_runtime.h>

class GPUTimer
{
private:
	cudaEvent_t start;
	cudaEvent_t stop;

public:
	GPUTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}

	~GPUTimer()
	{
		float elapsed = .0f;
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		std::cout << elapsed << " ms" << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
};
