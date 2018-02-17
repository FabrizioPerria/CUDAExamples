/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>

__global__ void findMin(const float* const input, float* const output)
{
	extern __shared__ float s_data[];
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	s_data[threadIdx.x] = input[index];

	syncthreads();

	for (unsigned int step = blockDim.x >> 1; step > 0; step >>= 1)
	{
		if(threadIdx.x < step)
			s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x + step]);
		syncthreads();
	}

	if(threadIdx.x == 0)
		output[blockIdx.x] = s_data[0];
}

__global__ void findMax(const float* const input, float* const output)
{
	extern __shared__ float s_data[];
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	s_data[threadIdx.x] = input[index];

	syncthreads();

	for (unsigned int step = blockDim.x >> 1; step > 0; step >>= 1)
	{
		if(threadIdx.x < step)
			s_data[threadIdx.x] = max(s_data[threadIdx.x], s_data[threadIdx.x + step]);
		syncthreads();
	}

	if(threadIdx.x == 0)
		output[blockIdx.x] = s_data[0];
}

__global__ void histogram(const float* const input, unsigned int* bins, const size_t numBins, const float min, const float range)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int bin = (input[index] - min) / range * numBins;

	if(bin >= numBins)
		bin--;

	atomicAdd(&(bins[bin]), 1);
}

__global__ void downSweep(unsigned int* const d_array, unsigned int numBins)
{
	extern __shared__ unsigned int s_dataBlelloch[];
	  int index = threadIdx.x;
	  s_dataBlelloch[index] = d_array[index];
	  __syncthreads();

	for(unsigned int step = 1; step < numBins; step <<= 1){
		if(index >= step)
			s_dataBlelloch[index] += s_dataBlelloch[index - step];
		__syncthreads();
	}
	 if (index == 0)
		 d_array[0] = 0;
	 else
		 d_array[index] = s_dataBlelloch[index - 1];
}

void findMinAndMaxLuminance(const float* const d_logLuminance, const size_t numRows, const size_t numCols, float& min_logLum, float& max_logLum)
{
	float* d_intermediate;
	float* d_min;
	float* d_max;
	cudaMalloc(&d_intermediate, numRows * sizeof(float));
	cudaMalloc(&d_min, sizeof(float));
	cudaMalloc(&d_max, sizeof(float));
	findMin<<<numRows, numCols, numCols * sizeof(float)>>>(d_logLuminance, d_intermediate);
	findMin<<<1, numRows, numRows * sizeof(float)>>>(d_intermediate, d_min);

	findMax<<<numRows, numCols, numCols * sizeof(float)>>>(d_logLuminance, d_intermediate);
	findMax<<<1, numRows, numRows * sizeof(float)>>>(d_intermediate, d_max);

	cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_intermediate);
	cudaFree(d_min);
	cudaFree(d_max);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

	findMinAndMaxLuminance(d_logLuminance, numRows, numCols, min_logLum, max_logLum);

	float range = max_logLum - min_logLum;

	histogram<<<numRows, numCols>>>(d_logLuminance, d_cdf, numBins, min_logLum, range);

	downSweep<<<1,numBins, numBins*sizeof(unsigned int)>>>(d_cdf, numBins);
}
