// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
	int index_x = threadIdx.x + blockIdx.x * blockDim.x;
	int index_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (index_x < numCols && index_y < numRows) {
		int index = numCols * index_y + index_x;

		uchar4 rgb_value = rgbaImage[index];
		greyImage[index] = .299f*rgb_value.x + .587f*rgb_value.y + .114f*rgb_value.z;
	}
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	const int blockWidth = 32;
	const dim3 blockSize(blockWidth, blockWidth, 1);
	const dim3 gridSize( numCols/blockWidth + 1, numRows/blockWidth + 1, 1);
	rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

}
