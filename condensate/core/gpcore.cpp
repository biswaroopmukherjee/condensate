
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>  
#include <chrono>

// CUDA+OPENGL includes
#include <GL/glew.h>
#include <algorithm>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <GL/freeglut.h>

#include "gpcore.hpp"
#include "render.hpp"
#include "initialize.hpp"
#include "defines.h"
#include "gp_kernels.h"
#include "helper_cudagl.h"



void evolve(int sizex, int sizey, cuDoubleComplex *arr) {
  int speed = 6000;
  int print = 5;

  printf("\n\n Starting GP... \n\n");

  render::startOpenGL();

  // // Create Wavefunction
  gpcore::initializecpy(arr);

  for( int a = 0; a < 200; a++ ) {
      double mult = exp(-(double)a / speed);
      multKernel(gpcore::devPsi, mult, DIM, DIM); 
      if (a % print == 0){
        glutMainLoopEvent();
      }
   }

  checkCudaErrors(cudaMemcpy(gpcore::hostPsi, gpcore::devPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyDeviceToHost));

  memcpy(arr, gpcore::hostPsi, sizeof(cuDoubleComplex) * DS);

  render::cleanup();

  printf("\n Done \n\n");

  // return gpcore::hostPsi;
}