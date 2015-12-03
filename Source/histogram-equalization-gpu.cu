#include "hist-equ-gpu.h"
#include <iostream>


void HistTest(PGM_IMG img_in) {
	int* h_hist = (int*) malloc(256 * sizeof(int));
	HistogramGPU(h_hist, img_in.img, img_in.h * img_in.w, 256);
	for(int i = 0; i < 256; i++) {
		std::cout<<h_hist[i]<<std::endl;
	}
}

void HistogramGPU(int * hist_out, unsigned char* img_in, int img_size, int nbr_bin) {
	int threadsPerBlock = 256;
	int numSMs = 192;

	// Initialize the histogram
	int* d_hist;
	cudaMalloc(&d_hist, nbr_bin * sizeof(int));
	MemsetGPU<<<numSMs * 32, threadsPerBlock>>>(d_hist, nbr_bin);

	//Click the counter
	unsigned char* imgData;
	size_t imgDataSize = img_size * sizeof(unsigned char);
	cudaMalloc(&imgData, imgDataSize);
	cudaMemcpy(imgData, img_in, imgDataSize, cudaMemcpyHostToDevice);
	HistogramGpuAction<<<numSMs * 32, threadsPerBlock>>>(d_hist, imgData, img_size);
	int* h_hist = (int*) malloc(256 * sizeof(int));

	//Copy back the memory
	cudaMemcpy(hist_out, d_hist, nbr_bin * sizeof(int), cudaMemcpyDeviceToHost);
}

void HistogramEqualizationGpu(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin) {
	int *lut = (int *)malloc(sizeof(int)*nbr_bin);
	int i, cdf, min, d;
	/* Construct the LUT by calculating the CDF */
	cdf = 0;
	min = 0;
	i = 0;
	while (min == 0){
		min = hist_in[i++];
	}
	d = img_size - min;	

	for (i = 0; i < nbr_bin; i++){
		cdf += hist_in[i];
		//lut[i] = (cdf - min)*(nbr_bin - 1)/d;
		lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
		if (lut[i] < 0){
			lut[i] = 0;
		}
	}

	/* Get the result image */
	for (i = 0; i < img_size; i++){
		if (lut[img_in[i]] > 255){
			img_out[i] = 255;
		}
		else{
			img_out[i] = (unsigned char)lut[img_in[i]];
		}
	}
}

__global__ void MemsetGPU(int* histOut, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		histOut[i] = 0;
	}
}

__global__ void HistogramGpuAction(int * histOut, unsigned char * imgIn, int imgSize) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < imgSize; i += blockDim.x * gridDim.x) {
		atomicAdd(&histOut[imgIn[i]], 1);
	}
}

__global__ void ArrayMin(int* dataIn, int* min, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		atomicMin(min, dataIn[i]);
	}
}