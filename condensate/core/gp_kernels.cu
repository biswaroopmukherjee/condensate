/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>  
#include <cufft.h>          // CUDA FFT Libraries

#include "defines.h"
#include "gp_kernels.h"

__device__
unsigned char clip(double x) {return x > 255 ? 255 : (x < 0 ? 0 : x); }


//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

__global__
void display_psi(uchar4 *d_out, cuDoubleComplex *devPsi, int w, int h) { 
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing
    d_out[i].x = 200 * devPsi[i].x;
    d_out[i].y = 200 * devPsi[i].x;
    d_out[i].z = 0;
    d_out[i].w = 255;
}

void kernelLauncher(uchar4 *d_out, cuDoubleComplex *devPsi, int w, int h) {
    const dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    const dim3 blockSize(TILEX, TILEY);
    display_psi<<<gridSize, blockSize>>>(d_out, devPsi, w, h);
}


__global__ 
void mult_psi(cuDoubleComplex *devPsi, double mult, int w, int h) {
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing
    devPsi[i].x = devPsi[i].x * mult;
    devPsi[i].y = devPsi[i].y * mult;
}

void multKernel(cuDoubleComplex *devPsi, double mult, int w, int h){
    const dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    const dim3 blockSize(TILEX, TILEY);
    mult_psi<<<gridSize, blockSize>>>(devPsi, mult, w, h);
}