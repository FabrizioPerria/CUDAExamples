#include <iostream>
#include "BitmapUtility.h"

const int DIM = 1024;

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b) {}
	__device__ float magnitude2( void ) { return r * r + i * i; }
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r+a.r, i+a.i);
	}
	__device__ int diverges() { return magnitude2() > 1000; }
};
struct cpuComplex {
	float r;
	float i;
};

cpuComplex juliaSetDictionary[] = {
		{-0.8, 0.156},
		{-0.7269, 0.1889},
		{-1.037, 0.17},
		{-0.52, 0.57},
		{0.295, 0.55},
		{-0.624, 0.435}
};

__device__ int julia(float realC, float imgC, int x, int y, int numRows, int numCols)
{
	const float scale = 1.5;
	float jx = scale * (1.0f - (float)(x / (float)(numCols >> 1)));
	float jy = scale * (1.0f - (float)(y / (float)(numRows >> 1)));
	cuComplex c(realC, imgC);
	cuComplex a(jx, jy);
	int i = 0;

	for (i=0; i<200; i++) {
		a = a * a + c;
		if (a.diverges())
			return 0;
	}

	return 1;
}

__global__ void runFractal(unsigned char* bitmapPtr, int numRows, int numCols, float realC, float imgC)
{
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (index_x < numCols && index_y < numRows) {
		int index = numCols * index_y + index_x;
		bitmapPtr[index*bytesPerPixel + 0] = 255 * julia(realC, imgC, index_x, index_y, numRows, numCols);
		bitmapPtr[index*bytesPerPixel + 1] = 0;
		bitmapPtr[index*bytesPerPixel + 2] = 0;
	}
}

int main(){
    int height = DIM;
    int width = DIM;
    unsigned char h_image[height][width][bytesPerPixel];
    unsigned char* d_image;
    cudaMalloc(&d_image, height * width * bytesPerPixel * sizeof(unsigned char));

    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    for(int i = 0; i < sizeof(juliaSetDictionary)/ sizeof(cpuComplex); ++i) {
		runFractal<<<gridSize, blockSize>>>(d_image, width, height, juliaSetDictionary[i].r, juliaSetDictionary[i].i);
		cudaMemcpy(h_image, d_image, height * width * bytesPerPixel, cudaMemcpyDeviceToHost);
		char name[30];
		sprintf(name, "fractal%d.bmp", i);
		generateBitmapImage((unsigned char*)h_image, height, width, name);
    }
    cudaFree(d_image);
}


