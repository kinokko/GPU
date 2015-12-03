#ifndef HIST_EQU_COLOR_GPU_H
#define HIST_EQU_COLOR_GPU_H

#include "hist-equ.h"
#include <cuda_runtime.h>

void HistTest(PGM_IMG img_in);
void HistogramGPU(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void HistogramEqualizationGpu(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);

void GenerateLUTGPU(int* dLut, int* dHistIn, int* dMin, int nbr_bin, int d);

__global__ void MemsetGPU(int * histOut, int histSize);
__global__ void HistogramGPUAction(int * histOut, unsigned char * imgIn, int imgSize);
__global__ void ArrayMin(int* dataIn, int* min, int size);
__global__ void CalculateD(int* min, int* d, int imgSize);
__global__ void GenerateLUTGPUAction(int* lut, int* histIn, int* min, int nbr_bin, int d);

#endif
