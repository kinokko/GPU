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
	HistogramGPUAction<<<numSMs * 32, threadsPerBlock>>>(d_hist, imgData, img_size);
	int* h_hist = (int*) malloc(256 * sizeof(int));

	//Copy back the memory
	cudaMemcpy(hist_out, d_hist, nbr_bin * sizeof(int), cudaMemcpyDeviceToHost);
}

void HistogramEqualizationGPU(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin) {
	int threadsPerBlock = 256;
	int numSMs = 192;

	int *lut = (int *)malloc(sizeof(int)*nbr_bin);
	int i, cdf, min, d;

	int* dLut;
	cudaMalloc(&dLut, sizeof(int)*nbr_bin);
	int* dHistIn;
	cudaMalloc(&dHistIn, sizeof(int)*nbr_bin);
	cudaMemcpy(dHistIn, hist_in, sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);
	int* dMin;
	cudaMalloc(&dMin, sizeof(int));
	int* dD;
	cudaMalloc(&dD, sizeof(int));
	ConstructLUTGPU(dLut, dHistIn, dMin, dD, nbr_bin, img_size, threadsPerBlock, numSMs);

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

void ConstructLUTGPU(int* dLut, int* dHistIn, int* dMin, int* dD, int nbr_bin, int imgSize, int threadsPerBlock, int numSMs) {
	ArrayMin<<<numSMs * 32, threadsPerBlock>>>(dHistIn, dMin, imgSize);
	CalculateD<<<numSMs * 32, threadsPerBlock>>>(dMin, dD, imgSize);
	GenerateLUTGPUAction<<<numSMs * 32, threadsPerBlock>>>(dLut, dHistIn, dMin, nbr_bin, dD, nbr_bin, img_size);
}

__global__ void MemsetGPU(int* histOut, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		histOut[i] = 0;
	}
}

__global__ void HistogramGPUAction(int * histOut, unsigned char * imgIn, int imgSize) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < imgSize; i += blockDim.x * gridDim.x) {
		atomicAdd(&histOut[imgIn[i]], 1);
	}
}

__global__ void ArrayMin(int* dataIn, int* min, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		atomicMin(min, dataIn[i]);
	}
}

__global__ void CalculateD(int* min, int* d, int imgSize) {
	*d = imgSize - *min;
}

__global__ void GenerateLUTGPUAction(int* lut, int* histIn, int* min, int* dD, int nbr_bin, int imgSize) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbr_bin; i += blockDim.x * gridDim.x) {
		int cdf = 0;
		for (int index = 0; index < i; index++) {
			cdf += histIn[index];
		}
		lut[i] = (int)(((float)cdf - *min) * 255 / *dD + 0.5);
		if (lut[i] < 0){
			lut[i] = 0;
		}
	}
}