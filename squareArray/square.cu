#include <iostream>
#include <memory>

__global__ void square1(float* out, float* in)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	float f = in[index];
	out[index] = f * f;
}

int main()
{
	const int N = 1024;
	std::unique_ptr<float[]> h_in(new float[N]);
	std::unique_ptr<float[]> h_out(new float[N]);

	for(int i = 0; i < N; ++i)
		h_in[i] = i;

	float *d_in = NULL;
	float *d_out = NULL;

	cudaMalloc(&d_in, N * sizeof(float));
	cudaMalloc(&d_out, N * sizeof(float));
	cudaMemcpy(d_in, h_in.get() , N * sizeof(float), cudaMemcpyHostToDevice);

	const int NUM_BLOCKS = 4;
	square1<<<NUM_BLOCKS, N/NUM_BLOCKS>>>(d_out, d_in);
	cudaDeviceSynchronize();

	cudaMemcpy(h_out.get(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; ++i) {
		std::cout << h_out[i];
		if(i && i%4 == 0)
			std::cout << std::endl;
		else
			std::cout << " ";
	}

	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
