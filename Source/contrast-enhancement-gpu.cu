#include "hist-equ-gpu.h"
#include <cuda_runtime.h>



//YUV Part


//End of YUV Part


//HSL Part

//End of HSL Part

//Helper 
__device__ unsigned char clip_rgb(int x)
{
	if (x > 255)
		return 255;
	if (x < 0)
		return 0;

	return (unsigned char)x;
}