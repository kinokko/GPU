#include "hist-equ-gpu.h"

__global__ void HistogramInitGpu(int * histOut, int blockSize) {
	int index = threadIdx.x + blockIdx.x * blockSize;
	histOut[index] = 0;
}

__global__ void HistogramGpu(int * histOut, unsigned char * imgIn, int imgSize, int blockSize, int totalBlock){
	int loop = 0;
	int index = threadIdx.x
			+ blockIdx.x * blockSize
			+ blockSize * totalBlock * blockSize;
	while (index < imgSize) {
		atomicAdd(&histOut[imgIn[index]], 1);
	}
}
