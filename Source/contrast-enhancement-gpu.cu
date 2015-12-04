#include <stdio.h>
#include <stdlib.h>
#include "hist-equ-gpu.h"
#include <cuda_runtime.h>



//YUV Part
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


//End of YUV Part


//HSL Part

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