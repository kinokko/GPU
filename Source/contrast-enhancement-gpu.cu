#include "hist-equ-gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


//YUV Part
void ContrastEnhancementGYUV(PPM_IMG img_in) {
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

	

}

__global__ void RGB2YUV_G(YUV_IMG img_out, PPM_IMG img_in, int img_size) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < img_size; i += blockDim.x * gridDim.x) {
		unsigned char r = img_in.img_r[i];
		unsigned char g = img_in.img_g[i];
		unsigned char b = img_in.img_b[i];

		unsigned char y = (unsigned char)(0.299*r + 0.587*g + 0.114*b);
		unsigned char cb = (unsigned char)(-0.169*r - 0.331*g + 0.499*b + 128);
		unsigned char cr = (unsigned char)(0.499*r - 0.418*g - 0.0813*b + 128);

		img_out.img_y[i] = y;
		img_out.img_u[i] = cb;
		img_out.img_v[i] = cr;
	}
}

//End of YUV Part


//HSL Part
template<class T>
T* CopyFromHostToDevice(T* p_data_in, bool reverse, int count = 1){
	T* p_data_out;
	size_t size = sizeof(T) * count;
	cudaMalloc(&p_data_out, size);
	cudaMemcpy(p_data_out, p_data_in, size, reverse ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
	return p_data_out;
}

template<class T>
T* MallocDataOnDevice(){

}


PPM_IMG* CopyRgbImageToDevice(PPM_IMG img_in, bool reverse = false){
	PPM_IMG* pd_img_out = CopyFromHostToDevice(&img_in, reverse);
	int count = img_in.h* img_in.w;
	//pointer members
	pd_img_out->img_r = CopyFromHostToDevice(img_in.img_r, reverse, count);
	pd_img_out->img_g = CopyFromHostToDevice(img_in.img_g, reverse, count);
	pd_img_out->img_b = CopyFromHostToDevice(img_in.img_b, reverse, count);
	return &img_in;
}

HSL_IMG* MallocHslImageOnDevice(int width, int height){
	HSL_IMG* pd_hsl_img;
	size_t size = width*height;
	cudaMalloc(&pd_hsl_img, sizeof(HSL_IMG));
	pd_hsl_img->width = width;
	pd_hsl_img->height = height;
	return pd_hsl_img;
}



PPM_IMG ContrastEnhancementGHSL(PPM_IMG img_in){
	HSL_IMG hsl_med;
	PPM_IMG result;

	unsigned char * l_equ;
	int hist[256];
	PPM_IMG* pd_img = CopyRgbImageToDevice(img_in);
	HSL_IMG* pd_hsl_img;

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

__global__ void RGB2HSL_G(HSL_IMG* pd_hsl_img_out, PPM_IMG* pd_img_in){

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