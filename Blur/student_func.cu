// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

#include "utils.h"
#include <stdio.h>

__device__
int clamp(int a, int b)
{
	return min(max(a, 0), b - 1);
}

__device__ void gaussian_blurNoShared(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	const int halfFilterWidth = filterWidth >> 1;

	if(index_x < numCols && index_y < numRows) {
		const int index = index_y * numCols + index_x;
		float sum = 0;
		for(int y = 0; y < filterWidth; ++y) {
			for(int x = 0; x < filterWidth; ++x) {
				int color_x = clamp(index_x + x - halfFilterWidth, numCols);
				int color_y = clamp(index_y + y - halfFilterWidth, numRows);
				float filterValue = filter[y * filterWidth + x];
				sum += filterValue * inputChannel[color_y * numCols + color_x];
			}
		}
		outputChannel[index] = sum;
	}
}

__device__ int isBottomSide(const int filterWidth)
{
	return threadIdx.y >= (blockDim.y - filterWidth + 1);
}

__device__ int isRightSide(const int filterWidth)
{
	return threadIdx.x >= (blockDim.x - filterWidth + 1);
}

__device__ int isBottomRightCorner(const int filterWidth)
{
	return threadIdx.x < (filterWidth - 1) && threadIdx.y < (filterWidth - 1);
}

__global__ void gaussian_blurWithShared(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
	extern __shared__ unsigned char s_array[];
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	const int index = index_y * numCols + index_x;

	const int halfFilterWidth = filterWidth >> 1;
	int positionToLoadFromX = clamp(index_x - halfFilterWidth, numCols);
	int positionToLoadFromY = clamp(index_y - halfFilterWidth, numRows);

	int blurredBlockDimension = (blockDim.x + filterWidth - 1);
	s_array[threadIdx.y* blurredBlockDimension + threadIdx.x] = inputChannel[positionToLoadFromY*numCols + positionToLoadFromX];

	int positionToLoadFromX_original = positionToLoadFromX;
	int positionToLoadFromY_original = positionToLoadFromY;

	if (isBottomSide(filterWidth)) {
		positionToLoadFromY = clamp(index_y + halfFilterWidth, numRows);
		s_array[(threadIdx.y + filterWidth - 1)*blurredBlockDimension + threadIdx.x] = inputChannel[positionToLoadFromY*numCols + positionToLoadFromX_original];
	}

	if (isRightSide(filterWidth)) {
		positionToLoadFromX = clamp(index_x + halfFilterWidth, numCols);
		s_array[threadIdx.y * blurredBlockDimension + threadIdx.x + filterWidth - 1] = inputChannel[positionToLoadFromY_original*numCols + positionToLoadFromX];
	}

	if(isBottomRightCorner(filterWidth)){
		positionToLoadFromX = clamp(index_x - halfFilterWidth + blockDim.x, numCols);
		positionToLoadFromY = clamp(index_y - halfFilterWidth + blockDim.y, numRows);
		s_array[(threadIdx.y + blockDim.y)*blurredBlockDimension + threadIdx.x + blockDim.x] = inputChannel[positionToLoadFromY*numCols + positionToLoadFromX];
	}

	__syncthreads();
	if(index_x < numCols && index_y < numRows) {
		float sum = 0;
		for(int y = 0; y < filterWidth; ++y) {
			for(int x = 0; x < filterWidth; ++x) {
				int color_x = threadIdx.x + x;
				int color_y = threadIdx.y + y;
				float filterValue = filter[y * filterWidth + x];
				float channelValue = (float)s_array[color_y * blurredBlockDimension + color_x];
				sum += filterValue * channelValue;
			}
		}
		outputChannel[index] = sum;
	}
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	if(index_x < numCols && index_y < numRows) {
		const int index = index_y * numCols + index_x;

		redChannel[index] = inputImageRGBA[index].x;
		greenChannel[index] = inputImageRGBA[index].y;
		blueChannel[index] = inputImageRGBA[index].z;
	}
}

	//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__ void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
	const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	if(index_x < numCols && index_y < numRows) {
		const int index = index_y * numCols + index_x;

		unsigned char red   = redChannel[index];
		unsigned char green = greenChannel[index];
		unsigned char blue  = blueChannel[index];

		uchar4 outputPixel = make_uchar4(red, green, blue, 255);

		outputImageRGBA[index] = outputPixel;
	}
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
	size_t channelSize = sizeof(unsigned char) * numRowsImage * numColsImage;
	checkCudaErrors(cudaMalloc(&d_red, channelSize));
	checkCudaErrors(cudaMalloc(&d_green, channelSize));
	checkCudaErrors(cudaMalloc(&d_blue, channelSize));

	size_t filterSize = sizeof(float) * filterWidth * filterWidth;
	checkCudaErrors(cudaMalloc(&d_filter, filterSize));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
	const int BLOCK_WIDTH = 32;
	const dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	const dim3 gridSize(numCols / BLOCK_WIDTH + 1, numRows / BLOCK_WIDTH + 1, 1);

	separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int shared_size = (blockSize.x + filterWidth - 1)*(blockSize.y + filterWidth - 1) * sizeof(unsigned char);

//	gaussian_blurNoShared<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
//	gaussian_blurNoShared<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
//	gaussian_blurNoShared<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
	gaussian_blurWithShared<<<gridSize, blockSize, shared_size>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
	gaussian_blurWithShared<<<gridSize, blockSize, shared_size>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
	gaussian_blurWithShared<<<gridSize, blockSize, shared_size>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_filter));
}
