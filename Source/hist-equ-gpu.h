#ifndef HIST_EQU_COLOR_GPU_H
#define HIST_EQU_COLOR_GPU_H
#include "hist-equ.h"
#include <cuda_runtime.h>
#include "Global.h"

PGM_IMG HistTest(PGM_IMG img_in);
void HistogramGPU(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
//void PreHistogramEqualizationGpu(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);

//void GenerateLUTGPU(int* dLut, int* dHistIn, int* dMin, int nbr_bin, int d);
void ConstructLUTGPU(int* dLut, int* dHistIn, int* dMin, int* dD, int nbr_bin, int imgSize);

__global__ void MemsetGPU(int * histOut, int histSize);
__global__ void HistogramGPUAction(int * histOut, unsigned char * imgIn, int imgSize);
__global__ void ArrayMin(int* dataIn, int* min, int size);
__global__ void CalculateD(int* d_out, int* hist_in, int* min_idx, int imgSize);
__global__ void GenerateLUTGPUAction(int* lut, int* histIn, int* min, int* dD, int nbr_bin, int imgSize);


// ---- Generate the new image based on the histogram ----
void HistogramEqualizationGPU(unsigned char * img_out, int * d_lut_in, unsigned char * d_img_in, int img_size);
__global__ void HistogramEqualizationGPUAction(unsigned char * d_img_out, int * d_lut_in, unsigned char * d_img_in, int imgSize);


//************************** Colorful World ************************//

//HSL Part
PPM_IMG ContrastEnhancementGHSL(PPM_IMG img_in);
__global__ void RGB2HSL_G(HSL_IMG pd_hsl_img_out, PPM_IMG d_img_in);
__global__ void HSL2RGB_G(PPM_IMG d_img_out, HSL_IMG d_img_in);
__device__ float Hue_2_RGB_G(float v1, float v2, float vH);
//End of HSL Part

//YUV Part
PPM_IMG ContrastEnhancementGYUV(PPM_IMG img_in);

__global__ void RGB2YUV_G(YUV_IMG d_img_out, PPM_IMG d_img_in, int img_size);
__global__ void YUV2RGB_G(PPM_IMG d_img_out, YUV_IMG d_img_in, int img_size);

//End of YUV Part




//Start Helper
__device__ unsigned char clip_rgb_gpu(int x);
//End helper

#endif
