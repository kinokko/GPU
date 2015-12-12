#ifndef SHARED
#include "hist-equ-gpu.h"


PGM_IMG HistTest(PGM_IMG img_in) {
	int nbr_bin = 256;
	int* d_hist;
	cudaMalloc(&d_hist, sizeof(int)*nbr_bin);

	// Copy image to device
	int img_size = img_in.h * img_in.w;
	unsigned char* d_img_in;
	size_t imgDataSize = img_size * sizeof(unsigned char);
	cudaMalloc(&d_img_in, imgDataSize);
	cudaMemcpy(d_img_in, img_in.img, imgDataSize, cudaMemcpyHostToDevice);

	// Get the histogram
	HistogramGPU(d_hist, d_img_in, img_size, nbr_bin);

	// init variables
	int* d_lut;
	cudaMalloc(&d_lut, sizeof(int)*nbr_bin);
	int* d_min;
	cudaMalloc(&d_min, sizeof(int));
	int* d_d;
	cudaMalloc(&d_d, sizeof(int));

	ConstructLUTGPU(d_lut, d_hist, d_min, d_d, nbr_bin, img_size);
	
	// ------------

	//Get output image data
	PGM_IMG result;
	result.w = img_in.w;
	result.h = img_in.h;
	result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	HistogramEqualizationGPU(result.img, d_lut, d_img_in, img_in.h * img_in.w);

	cudaFree(d_img_in);
	cudaFree(d_lut);
	cudaFree(d_min);
	cudaFree(d_d);

	return result;
}

void HistogramGPU(int* d_hist_out, unsigned char* d_img_in, int img_size, int nbr_bin) {
	int threadsPerBlock = 512;
	int numSMs = 192;

	// Initialize the histogram
	MemsetGPU<<<1, 256>>>(d_hist_out, nbr_bin);

	//Count the color
	HistogramGPUAction<<<BLOCKPERGRID, THREADSPERBLOCK>>>(d_hist_out, d_img_in, img_size);
}



void ConstructLUTGPU(int* d_lut, int* d_histIn, int* d_min_idx, int* d_d, int nbr_bin, int imgSize) {
	cudaMemcpy(d_min_idx, &nbr_bin, sizeof(int), cudaMemcpyHostToDevice);
	ArrayMin<<<1, 256>>>(d_histIn, d_min_idx, nbr_bin);
	CalculateD<<<1,1>>>(d_d, d_histIn, d_min_idx, imgSize);
	GenerateLUTGPUAction<<<1, 256>>>(d_lut, d_histIn, d_min_idx, d_d, nbr_bin, imgSize);
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
__global__ void ArrayMin(int* dataIn, int* min_idx, int size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		if(dataIn[i] != 0) {
			atomicMin(min_idx, i);
		}
	}
}


__global__ void CalculateD(int* d_out, int* hist_in, int* min_idx, int imgSize) {
	*d_out = imgSize - hist_in[*min_idx];
}

__global__ void GenerateLUTGPUAction(int* lut, int* histIn, int* min, int* d_d, int nbr_bin, int imgSize) {
	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbr_bin; i += blockDim.x * gridDim.x) {
		int cdf = 0;
		for (int index = 0; index <= i; index++) {
			cdf += histIn[index];
		}
		lut[i] = (int)(((float)cdf - histIn[*min]) * 255 / *d_d + 0.5);
		if (lut[i] < 0){
			lut[i] = 0;
		}
	}
}

// ---- Generate the new image based on the histogram ----
void HistogramEqualizationGPU(unsigned char * img_out, int * d_lut_in , unsigned char * d_img_in, int img_size){
	int threadsPerBlock = 512;
	int numSMs = 192;

	// Prepare device memory for output image
	unsigned char * d_img_out;
	cudaMalloc(&d_img_out, img_size);
	HistogramEqualizationGPUAction <<<numSMs * 32, threadsPerBlock >>> (d_img_out, d_lut_in, d_img_in, img_size);
	cudaDeviceSynchronize();
	// Copy output image back to host memory
	cudaMemcpy(img_out, d_img_out, img_size, cudaMemcpyDeviceToHost);
	cudaFree(d_img_out);
}

__global__ void HistogramEqualizationGPUAction(unsigned char * d_img_out, int * d_lut_in, unsigned char * d_img_in, int imgSize) {
#ifdef GOODSHARE
	__shared__ int loc_lut[256];
	__syncthreads();
	for (int i = threadIdx.x; i < 256; i += blockDim.x) {
		loc_lut[i] = d_lut_in[i];
	}
	__syncthreads();
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < imgSize; i += blockDim.x * gridDim.x) {
		unsigned char pixel = d_img_in[i];
		if (loc_lut[pixel] > 255){
			d_img_out[i] = 255;	
		}
		else{
			d_img_out[i] = (unsigned char)loc_lut[pixel];
		}
	}
#else
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < imgSize; i += blockDim.x * gridDim.x) {
		if (d_lut_in[d_img_in[i]] > 255){
			d_img_out[i] = 255;	
		}
		else{
			d_img_out[i] = (unsigned char)d_lut_in[d_img_in[i]];
		}
	}
#endif
}

#endif