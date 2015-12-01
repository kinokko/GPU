#include "hist-equ-gpu.cuh"

__global__ void HistogramInitGpu (int * histOut, int offSet) {
	int index = threadIdx.x + blockIdx.x + offSet;
	histOut
}

__global__ void HistogramGpu(int * histOut, unsigned char * imgIn, int imgSize, int blockSize, int totalBlock){
	int loop = 0;
	int index = threadIdx.x
			+ blockIdx.x * blockSize
			+ blockSize * totalBlock * blockSize;
	while (index < imgSize) {
		atomicAdd(histOut[imgIn[index]], 1);
	}
}
