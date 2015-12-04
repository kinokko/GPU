#include "hist-equ-gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


//YUV Part
PPM_IMG ContrastEnhancementGYUV(PPM_IMG img_in) {
	int img_size = img_in.w * img_in.h;
	int img_data_size = img_size * sizeof(unsigned char);

	//copy img data to device memory
	PPM_IMG d_img_rgb;
	d_img_rgb.h = img_in.h;
	d_img_rgb.w = img_in.w;
	cudaMalloc(&d_img_rgb.img_r, img_data_size);
	cudaMalloc(&d_img_rgb.img_g, img_data_size);
	cudaMalloc(&d_img_rgb.img_b, img_data_size);
	cudaMemcpy(d_img_rgb.img_r, img_in.img_r, img_data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_rgb.img_g, img_in.img_g, img_data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img_rgb.img_b, img_in.img_b, img_data_size, cudaMemcpyHostToDevice);

	// init a device memory for yuv img
	YUV_IMG d_img_yuv;
	d_img_yuv.h = img_in.h;
	d_img_yuv.w = img_in.w;
	cudaMalloc(&d_img_yuv.img_y, img_data_size);
	cudaMalloc(&d_img_yuv.img_u, img_data_size);
	cudaMalloc(&d_img_yuv.img_v, img_data_size);

	
	RGB2YUV_G<<<BLOCKPERGRID, THREADSPERBLOCK>>>(d_img_yuv, d_img_rgb, img_size);


	//hist
	int nbr_bin = 256;
	int* d_hist;
	cudaMalloc(&d_hist, sizeof(int)*nbr_bin);
	HistogramGPU(d_hist, d_img_yuv.img_y, img_size, nbr_bin);

	//hist_equ
	int* d_lut;
	cudaMalloc(&d_lut, sizeof(int)*nbr_bin);
	int* d_min;
	cudaMalloc(&d_min, sizeof(int));
	int* d_d;
	cudaMalloc(&d_d, sizeof(int));
	ConstructLUTGPU(d_lut, d_hist, d_min, d_d, nbr_bin, img_size);

	unsigned char* proceed_img;
	cudaMalloc(&proceed_img, img_data_size);
	HistogramEqualizationGPUAction<<<BLOCKPERGRID, THREADSPERBLOCK>>>(proceed_img, d_lut, d_img_yuv.img_y, img_size);
	cudaFree(d_img_yuv.img_y);
	d_img_yuv.img_y = proceed_img;

	YUV2RGB_G<<<BLOCKPERGRID, THREADSPERBLOCK>>>(d_img_rgb, d_img_yuv, img_size);

	PPM_IMG h_img_rgb;
	h_img_rgb.h = img_in.h;
	h_img_rgb.w = img_in.w;
	h_img_rgb.img_r = (unsigned char*)malloc(img_data_size);
	h_img_rgb.img_g = (unsigned char*)malloc(img_data_size);
	h_img_rgb.img_b = (unsigned char*)malloc(img_data_size);
	cudaMemcpy(h_img_rgb.img_r, d_img_rgb.img_r, img_data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_img_rgb.img_g, d_img_rgb.img_g, img_data_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_img_rgb.img_b, d_img_rgb.img_b, img_data_size, cudaMemcpyDeviceToHost);
	return h_img_rgb;
}

__global__ void RGB2YUV_G(YUV_IMG d_img_out, PPM_IMG d_img_in, int img_size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_size; i += blockDim.x * gridDim.x) {
		unsigned char r = d_img_in.img_r[i];
		unsigned char g = d_img_in.img_g[i];
		unsigned char b = d_img_in.img_b[i];

		unsigned char y = (unsigned char)(0.299*r + 0.587*g + 0.114*b);
		unsigned char cb = (unsigned char)(-0.169*r - 0.331*g + 0.499*b + 128);
		unsigned char cr = (unsigned char)(0.499*r - 0.418*g - 0.0813*b + 128);

		d_img_out.img_y[i] = y;
		d_img_out.img_u[i] = cb;
		d_img_out.img_v[i] = cr;
	}
}

__global__ void YUV2RGB_G(PPM_IMG d_img_out, YUV_IMG d_img_in, int img_size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_size; i += blockDim.x * gridDim.x) {
		int y = (int)d_img_in.img_y[i];
		int cb = (int)d_img_in.img_u[i] - 128;
		int cr = (int)d_img_in.img_v[i] - 128;

		int rt = (int)(y + 1.402*cr);
		int gt = (int)(y - 0.344*cb - 0.714*cr);
		int bt = (int)(y + 1.772*cb);

		d_img_out.img_r[i] = clip_rgb_gpu(rt);
		d_img_out.img_g[i] = clip_rgb_gpu(gt);
		d_img_out.img_b[i] = clip_rgb_gpu(bt);
	}
}

//End of YUV Part


//HSL Part
PPM_IMG ContrastEnhancementGHSL(PPM_IMG img_in){
	HSL_IMG hsl_med;
	PPM_IMG result;

	unsigned char * l_equ;
	int hist[256];

	hsl_med = rgb2hsl(img_in);
	l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

	histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
	histogram_equalization(l_equ, hsl_med.l, hist, hsl_med.width*hsl_med.height, 256);

	free(hsl_med.l);
	hsl_med.l = l_equ;

	result = hsl2rgb(hsl_med);
	free(hsl_med.h);
	free(hsl_med.s);
	free(hsl_med.l);
	return result;
}
//End of HSL Part

//Helper 
__device__ unsigned char clip_rgb_gpu(int x)
{
	if (x > 255)
		return 255;
	if (x < 0)
		return 0;

	return (unsigned char)x;
}