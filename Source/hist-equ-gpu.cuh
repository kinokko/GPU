#ifndef HIST_EQU_COLOR_GPU_H
#define HIST_EQU_COLOR_GPU_H

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);

#endif
