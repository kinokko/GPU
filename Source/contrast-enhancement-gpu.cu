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
template<class T>
T* CopyFromHostToDevice(T* p_data_in, bool reverse, int count = 1){
	T* p_data_out;
	size_t size = sizeof(T) * count;
	if (reverse){
		p_data_out = (T*)malloc(size);
	}
	else{
		cudaMalloc(&p_data_out, size);
	}
	cudaMemcpy(p_data_out, p_data_in, size, reverse ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
	return p_data_out;
}

template<class T>
T* MallocDataOnDevice(){

}


PPM_IMG CreateCopyRgbImageToDevice(PPM_IMG img_in, bool reverse = false){
	PPM_IMG d_img_out;
	d_img_out.w = img_in.w;
	d_img_out.h = img_in.h;
	int img_size = img_in.h* img_in.w;
	d_img_out.img_r = CopyFromHostToDevice(img_in.img_r, reverse, img_size);
	d_img_out.img_g = CopyFromHostToDevice(img_in.img_g, reverse, img_size);
	d_img_out.img_b = CopyFromHostToDevice(img_in.img_b, reverse, img_size);
	return d_img_out;
}

HSL_IMG CreateCopyHslImageToDevice(HSL_IMG img_in, bool reverse = false){
	HSL_IMG d_img_out;
	d_img_out.width = img_in.width;
	d_img_out.height = img_in.height;
	int img_size = img_in.height* img_in.width;
	d_img_out.h = CopyFromHostToDevice(img_in.h, reverse, img_size);
	d_img_out.s = CopyFromHostToDevice(img_in.s, reverse, img_size);
	d_img_out.l = CopyFromHostToDevice(img_in.l, reverse, img_size);
	return d_img_out;
}

HSL_IMG MallocHslImageOnDevice(int width, int height){
	HSL_IMG d_hsl_img;
	size_t img_size = width*height;
	d_hsl_img.width = width;
	d_hsl_img.height = height;
	cudaMalloc(&d_hsl_img.h, sizeof(float)* img_size);
	cudaMalloc(&d_hsl_img.s, sizeof(float)* img_size);
	cudaMalloc(&d_hsl_img.l, sizeof(unsigned char)* img_size);
	return d_hsl_img;
}


PPM_IMG ContrastEnhancementGHSL(PPM_IMG img_in){
	HSL_IMG hsl_med;
	PPM_IMG result;

	unsigned char * l_equ;
	int hist[256];
	PPM_IMG d_img = CreateCopyRgbImageToDevice(img_in);
	PPM_IMG test = CreateCopyRgbImageToDevice(d_img, true);
	HSL_IMG d_hsl_img = MallocHslImageOnDevice(img_in.w, img_in.h);
	//hsl_med = rgb2hsl(img_in);
	//hsl_med = rgb2hsl(CopyRgbImageToDevice(d_img, true));
	RGB2HSL_G<<<6144,512>>>(d_hsl_img, d_img);
	hsl_med = CreateCopyHslImageToDevice(d_hsl_img, true);
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

__global__ void RGB2HSL_G(HSL_IMG d_hsl_img, PPM_IMG d_img_in){


	//printf("%d,%d", d_hsl_img.height, d_hsl_img.width);
	float H, S, L;
	int size = d_img_in.h*d_img_in.w;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
		float var_r = ((float)d_img_in.img_r[i] / 255);//Convert RGB to [0,1]
		float var_g = ((float)d_img_in.img_g[i] / 255);
		float var_b = ((float)d_img_in.img_b[i] / 255);
		float var_min = (var_r < var_g) ? var_r : var_g;
		var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
		float var_max = (var_r > var_g) ? var_r : var_g;
		var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
		float del_max = var_max - var_min;               //Delta RGB value

		L = (var_max + var_min) / 2;
		if (del_max == 0)//This is a gray, no chroma...
		{
			H = 0;
			S = 0;
		}
		else                                    //Chromatic data...
		{
			if (L < 0.5)
				S = del_max / (var_max + var_min);
			else
				S = del_max / (2 - var_max - var_min);

			float del_r = (((var_max - var_r) / 6) + (del_max / 2)) / del_max;
			float del_g = (((var_max - var_g) / 6) + (del_max / 2)) / del_max;
			float del_b = (((var_max - var_b) / 6) + (del_max / 2)) / del_max;
			if (var_r == var_max){
				H = del_b - del_g;
			}
			else{
				if (var_g == var_max){
					H = (1.0 / 3.0) + del_r - del_b;
				}
				else{
					H = (2.0 / 3.0) + del_g - del_r;
				}
			}

		}

		if (H < 0)
			H += 1;
		if (H > 1)
			H -= 1;

		d_hsl_img.h[i] = H;
		d_hsl_img.s[i] = S;
		d_hsl_img.l[i] = (unsigned char)(L * 255);
	}

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