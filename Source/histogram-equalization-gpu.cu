#include "hist-equ-gpu.h"

__global__ void HistogramInitGpu(int * histOut, int histSize) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < histSize; i += blockDim.x * gridDim.x) {
		histOut[i] = 0;
	}
}

__global__ void HistogramGpu(int * histOut, unsigned char * imgIn, int imgSize, int threadsPerBlock, int blocksPerGrid){
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < histSize; i += blockDim.x * gridDim.x) {
		atomicAdd(&histOut[imgIn[index]], 1);
	}
}
