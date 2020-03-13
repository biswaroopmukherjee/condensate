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

__constant__ double gDenConst = 6.6741e-40;//Evaluted in MATLAB: N*4*HBAR*HBAR*PI*(4.67e-9/mass)*sqrt(mass*(omegaZ)/(2*PI*HBAR))


// *****************************
// Useful functions
//******************************
__device__
unsigned char clip(double x) {return x > 255 ? 255 : (x < 0 ? 0 : x); }

__device__
uchar4 viridis(double value) {
    uchar4 result;
    result.x = clip(255 * ( 2.854 * pow(value, 3) - 2.098 * pow(value, 2) + 0.037 * value + 0.254));
    result.y = clip(255 * (-0.176 * pow(value, 3) - 0.167 * pow(value, 2) + 1.243 * value + 0.016));
    result.z = clip(255 * ( 0.261 * pow(value, 3) - 1.833 * pow(value, 2) + 1.275 * value + 0.309));
    result.w = 255;
    return result;
}

__device__
uchar4 inferno(double value) {
    uchar4 result;
    result.x = clip(255 * (-1.760 * pow(value, 3) + 1.487  * pow(value, 2) + 1.223 * value - 0.034));
    result.y = clip(255 * ( 0.284 * pow(value, 3) - 0.827  * pow(value, 2) - 0.086 * value + 0.026));
    result.z = clip(255 * ( 7.533 * pow(value, 3) - 11.435 * pow(value, 2) + 4.603 * value - 0.096));
    result.w = 255;
    return result;
}


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

// Calculate the magnitude squared of a complex number
__host__ __device__
double complexMagnitudeSquared(cuDoubleComplex in){
	return in.x*in.x + in.y*in.y;
}



// *****************************
// GPU Kernels
//******************************

// Handles display mapping from cuDoubleComplex to uchar4
__global__
void display_psi(uchar4 *d_out, cuDoubleComplex *devPsi, double scale, int w, int h) { 
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing
    double mag = complexMagnitudeSquared(devPsi[i]);
    d_out[i] = viridis(mag/scale);
}

void colormapKernelLauncher(uchar4 *d_out, cuDoubleComplex *devPsi, double scale, int w, int h) {
    const dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    const dim3 blockSize(TILEX, TILEY);
    display_psi<<<gridSize, blockSize>>>(d_out, devPsi, scale, w, h);
}



// Multiply the wavefunction with a real-valued scalar
__global__ 
void realmult_psi(cuDoubleComplex *devPsi, double mult, int w, int h) {
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing
    devPsi[i].x = devPsi[i].x * mult;
    devPsi[i].y = devPsi[i].y * mult;
}

void multKernelLauncher(cuDoubleComplex *devPsi, double mult, int w, int h){
    const dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    const dim3 blockSize(TILEX, TILEY);
    realmult_psi<<<gridSize, blockSize>>>(devPsi, mult, w, h);
}



// Realspace evolution 
__global__
void realevolve_psi(cuDoubleComplex *devPsi, cuDoubleComplex *devExpPotential, cuDoubleComplex *out, double dt, double useReal, double cooling, int w, int h){
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing

    cuDoubleComplex tPotential = devExpPotential[i];
    cuDoubleComplex tPsi = devPsi[i];
    double gn = gDenConst * complexMagnitudeSquared(tPsi) * (dt / (2*HBAR));
    cuDoubleComplex expgn;
    expgn.x = exp( -gn * cooling) * cos( -gn * useReal);
    expgn.y = exp( -gn * cooling) * sin( -gn * useReal);

    cuDoubleComplex realspaceUnitary = cuCmul(tPotential, expgn);
    out[i] = cuCmul(realspaceUnitary, tPsi);
}

void realspaceKernelLauncher(cuDoubleComplex *devPsi, cuDoubleComplex *devExpPotential, cuDoubleComplex *out, double dt, double useReal, double cooling, int w, int h){
    const dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    const dim3 blockSize(TILEX, TILEY);
    realevolve_psi<<<gridSize, blockSize>>>(devPsi, devExpPotential, out, dt, useReal, cooling, w, h);
}


// Momentum Space evolution 
__global__
void momentumevolve_psi(cuDoubleComplex *devPsi, cuDoubleComplex *devExpKinetic, cuDoubleComplex *out, int w, int h){
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing

    cuDoubleComplex tKinetic = devExpKinetic[i];
    cuDoubleComplex tPsi = devPsi[i];
    out[i] = cuCmul(tKinetic, tPsi);
}

void momentumspaceKernelLauncher(cuDoubleComplex *devPsi, cuDoubleComplex *devExpKinetic, cuDoubleComplex *out, int w, int h){
    const dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    const dim3 blockSize(TILEX, TILEY);
    momentumevolve_psi<<<gridSize, blockSize>>>(devPsi, devExpKinetic, out, w, h);
}


// Density psi
__global__ 
void density_psi(cuDoubleComplex *devPsi, double *density, int w, int h) {
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing

    density[i] = complexMagnitudeSquared(devPsi[i]);
}
//Normalization
__global__ 
void scalarDiv_wfcNorm(cuDoubleComplex *in, double dr, double* pSum, cuDoubleComplex *out, int w, int h){
    const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    const int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    if ((tidx >= w) || (tidy >= h)) return; // Check if in bounds
    const int i = tidx + tidy * w; // 1D indexing

    cuDoubleComplex result;
    double norm = sqrt((pSum[0])*dr);
    result.x = (in[i].x/norm);
    result.y = (in[i].y/norm);
    out[i] = result;
}
// Indexing for parallel summation
__device__ unsigned int getGid3d3d(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                  + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.y * blockDim.x)
                   + (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
    return threadId;
}
/**
 * Routine for parallel summation. Can be looped over from host.
 */
 __global__ void multipass(double* input, double* output){
    unsigned int tid = threadIdx.x + threadIdx.y*blockDim.x
                       + threadIdx.z * blockDim.x * blockDim.y;
    unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x
                       + gridDim.x * gridDim.y * blockIdx.z;

    //unsigned int tid = getTid3d3d();
    //unsigned int bid = getBid3d3d();
    // printf("bid0=%d\n",bid);

    unsigned int gid = getGid3d3d();
    extern __shared__ double sdatad[];
    sdatad[tid] = input[gid];
    __syncthreads();

    for(int i = blockDim.x>>1; i > 0; i>>=1){
        if(tid < i){
            sdatad[tid] += sdatad[tid + i];
        }
        __syncthreads();
    }
    if(tid==0){
        output[bid] = sdatad[0];
    }
}
/*
 * General-purpose summation of an array on the gpu, storing the result in the first element
*/
void gpuReduce(double* data, int length, int threadCount) {
    dim3 block(length / threadCount, 1, 1);
    dim3 threads(threadCount, 1, 1);

    while((double)length/threadCount > 1.0){
        multipass<<<block,threads,threadCount*sizeof(double)>>>(&data[0],
                                                                &data[0]);
        length /= threadCount;
        block = (int) ceil((double)length/threadCount);
    }
    multipass<<<1,length,threadCount*sizeof(double)>>>(&data[0],
                                                       &data[0]);
}

void parSum(cuDoubleComplex *devPsi, double *density, double dx, int w, int h){

    int DS = w * h;
    double dg = dx * dx;

    dim3 gridSize (iDivUp(w, TILEX), iDivUp(h, TILEY));
    dim3 grid_tmp(DS, 1, 1);
    dim3 threads(TILEX, TILEY);
    dim3 block(grid_tmp.x/threads.x, 1, 1);

    density_psi<<<gridSize, threads>>>(devPsi, density, w, h);

    gpuReduce(density, grid_tmp.x, threads.x);
/*
    // Writing out in the parSum Function (not recommended, for debugging)
    double *sum;
    sum = (double *) malloc(sizeof(double)*gsize);
    cudaMemcpy(sum,density,sizeof(double)*gsize,
               cudaMemcpyDeviceToHost);
    std::cout << (sum[0]) << '\n';
*/
    scalarDiv_wfcNorm<<<gridSize, threads>>>(devPsi, dg, density, devPsi, w, h);
}