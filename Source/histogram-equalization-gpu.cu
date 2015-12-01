#include "hist-equ-gpu.cuh"

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
	int x = blockIdx.x + threadIdx.x;
}
