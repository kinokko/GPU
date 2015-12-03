#include "hist-equ-gpu.h"
#include <iostream>

PGM_IMG HistTest(PGM_IMG img_in) {
	int* h_hist = (int*) malloc(256 * sizeof(int));
	HistogramGPU(h_hist, img_in.img, img_in.h * img_in.w, 256);

	// ------------ To Be replaced By GPU LUT generation
	int *lut = (int *)malloc(sizeof(int) * 256);
	int i, cdf, min, d;
	/* Construct the LUT by calculating the CDF */
	cdf = 0;
	min = 0;
	i = 0;
	while (min == 0){
		min = h_hist[i++];
	}
	d = img_in.h * img_in.w - min;
	for (i = 0; i < 256; i++){
		cdf += h_hist[i];
		//lut[i] = (cdf - min)*(nbr_bin - 1)/d;
		lut[i] = (int)(((float)cdf - min) * 255 / d + 0.5);
		if (lut[i] < 0){
			lut[i] = 0;
		}
	}

	int* d_lut;
	cudaMalloc(&d_lut, 256 * sizeof(int));
	cudaMemcpy(d_lut, lut, 256 * sizeof(int), cudaMemcpyHostToDevice);

	unsigned char* d_img_in;
	size_t imgDataSize = img_in.h * img_in.w * sizeof(unsigned char);
	cudaMalloc(&d_img_in, imgDataSize);
	cudaMemcpy(d_img_in, img_in.img, imgDataSize, cudaMemcpyHostToDevice);
	// ------------

	//Get output image data
	PGM_IMG result;
	result.w = img_in.w;
	result.h = img_in.h;
	result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	HistogramEqualizationGPU(result.img, d_lut, d_img_in, img_in.h * img_in.w);

	return result;
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

__global__ void MemsetGPU(int* histOut, int nbr_bin) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbr_bin; i += blockDim.x * gridDim.x) {
		histOut[i] = 0;
	}
}

__global__ void HistogramGpuAction(int * histOut, unsigned char * imgIn, int imgSize) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < imgSize; i += blockDim.x * gridDim.x) {
		atomicAdd(&histOut[imgIn[i]], 1);
	}
}

// ---- Generate the new image based on the histogram ----
void HistogramEqualizationGPU(unsigned char * img_out, int * d_lut_in , unsigned char * d_img_in, int img_size){
	int threadsPerBlock = 256;
	int numSMs = 192;

	// Prepare device memory for output image
	unsigned char * d_img_out;
	cudaMalloc(&d_img_out, img_size);
	HistogramEqualizationGPUAction <<<numSMs * 32, threadsPerBlock >>> (d_img_out, d_lut_in, d_img_in, img_size);
	cudaDeviceSynchronize();
	// Copy output image back to host memory
	cudaMemcpy(img_out, d_img_out, img_size, cudaMemcpyDeviceToHost);

}

__global__ void HistogramEqualizationGPUAction(unsigned char * d_img_out, int * d_lut_in, unsigned char * d_img_in, int imgSize) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < imgSize; i += blockDim.x * gridDim.x) {
		if (d_lut_in[d_img_in[i]] > 255){
			d_img_out[i] = 255;	
		}
		else{
			d_img_out[i] = (unsigned char)d_lut_in[d_img_in[i]];
		}
	}
}
