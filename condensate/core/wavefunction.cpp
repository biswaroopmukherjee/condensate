#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>  
#include <algorithm>

#include "helper_cudagl.h"
#include "defines.h"
#include "wavefunction.hpp"
#include "gp_kernels.h"


void Wavefunction::Initialize(cuDoubleComplex *arr)
{
    // Create Wavefunction
    hostPsi = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * DS);
    memcpy( hostPsi, arr, sizeof(cuDoubleComplex) * DS);
    // Allocate and initialize device data
    checkCudaErrors(cudaMalloc((void **)&devPsi, sizeof(cuDoubleComplex) * DIM * DIM));
    checkCudaErrors(cudaMemcpy(devPsi, hostPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));

}

void Wavefunction::Step(double mult)
{
    multKernel(devPsi, mult, DIM, DIM);
}


void Wavefunction::MapColors(uchar4 *d_out)
{
    kernelLauncher(d_out, devPsi, DIM, DIM);
}


void Wavefunction::ExportToVariable(cuDoubleComplex *arr)
{
    checkCudaErrors(cudaMemcpy(hostPsi, devPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyDeviceToHost));
    memcpy(arr, hostPsi, sizeof(cuDoubleComplex) * DS);
}


void Wavefunction::Cleanup()
{
    free(hostPsi);
    checkCudaErrors(cudaFree(devPsi));
    checkCudaErrors(cudaDeviceReset());
}