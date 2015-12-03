#ifndef HIST_EQU_COLOR_GPU_H
#define HIST_EQU_COLOR_GPU_H

#include "hist-equ.h"
#include <cuda_runtime.h>

PGM_IMG HistTest(PGM_IMG img_in);
void HistogramGPU(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void HistogramEqualizationGpu(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);


__global__ void MemsetGPU(int * histOut, int histSize);
__global__ void HistogramGpuAction(int * histOut, unsigned char * imgIn, int imgSize);



// ---- Generate the new image based on the histogram ----
void HistogramEqualizationGPU(unsigned char * img_out, int * d_lut_in, unsigned char * d_img_in, int img_size);
__global__ void HistogramEqualizationGPUAction(unsigned char * d_img_out, int * d_lut_in, unsigned char * d_img_in, int imgSize);

#endif
