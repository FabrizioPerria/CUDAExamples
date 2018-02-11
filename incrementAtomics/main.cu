#include <stdio.h>
#include "gputimer.h"

#define NUM_THREADS 1000000
#define ARRAY_SIZE  100

#define BLOCK_WIDTH 1000

void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < size; i++)  { printf("%d ", array[i]); }
    printf("}\n");
}

__global__ void increment1(int *g){
    //10^6 threads incrementing 10^6 elements non-atomically
    int i = threadIdx.x;
    g[i]++;
}

__global__ void increment2(int *g){
    //10^6 threads incrementing 10^6 elements atomically
    int i = threadIdx.x;
    atomicAdd(&g[i], 1);
}

__global__ void increment3(int *g){
	//10^6 threads incrementing 100 elements non-atomically
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	g[i]++;
}

__global__ void increment4(int *g)
{
	//10^6 threads incrementing 100 elements atomically
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	atomicAdd(& g[i], 1);
}

int main(int argc,char **argv)
{
    GpuTimer timer;
    printf("%d total threads in %d blocks writing into %d array elements\n",
           NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // declare, allocate, and zero out GPU memory
    int * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void *) d_array, 0, ARRAY_BYTES);

    // launch the kernel - comment out one of these
    timer.Start();
    increment1<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_BYTES);
    memset(h_array, 0, ARRAY_BYTES);
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    timer.Start();
    increment2<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_BYTES);
    memset(h_array, 0, ARRAY_BYTES);
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    timer.Start();
    increment3<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_BYTES);
    memset(h_array, 0, ARRAY_BYTES);
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    timer.Start();
    increment4<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_BYTES);
    memset(h_array, 0, ARRAY_BYTES);
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    timer.Start();
    increment4<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH*10>>>(d_array);
    timer.Stop();
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_BYTES);
    memset(h_array, 0, ARRAY_BYTES);
    printf("Time elapsed = %g ms\n", timer.Elapsed());

    // free GPU memory allocation and exit
    cudaFree(d_array);
    return 0;
}
