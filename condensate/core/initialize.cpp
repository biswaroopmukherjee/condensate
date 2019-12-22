
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>  
#include <algorithm>

#include "helper_cudagl.h"
#include "defines.h"
#include "initialize.hpp"

namespace gpcore {

    cuDoubleComplex *devPsi, *hostV, *hostPsi;
    int wWidth = std::max(512, DIM);
    int wHeight = std::max(512, DIM);

    void initPsi(cuDoubleComplex *p, int dx, int dy) {
        int i, j;

        for (i = 0; i < dy; i++) {
            for (j = 0; j < dx; j++) {
            p[i * dx + j].x = exp(-( pow( (i-200.)/200., 2. ) + pow( (j-200.)/200., 2. ) ) );
            p[i * dx + j].y = exp(-( pow( (i-200.)/200., 2. ) + pow( (j-200.)/200., 2. ) ) );
            }
        }
    }

    void initialize(){
        // Create Wavefunction
        hostPsi = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * DS);
        memset(hostPsi, 0, sizeof(cuDoubleComplex) * DS);
        initPsi(hostPsi, DIM, DIM);
        // Allocate and initialize device data
        checkCudaErrors(cudaMalloc((void **)&devPsi, sizeof(cuDoubleComplex) * DIM * DIM));
        checkCudaErrors(cudaMemcpy(devPsi, hostPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));

    }

    void initializecpy(cuDoubleComplex *arr){
        // Create Wavefunction
        hostPsi = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * DS);
        memcpy( hostPsi, arr, sizeof(cuDoubleComplex) * DS);
        // Allocate and initialize device data
        checkCudaErrors(cudaMalloc((void **)&devPsi, sizeof(cuDoubleComplex) * DIM * DIM));
        checkCudaErrors(cudaMemcpy(devPsi, hostPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));

    }

}

