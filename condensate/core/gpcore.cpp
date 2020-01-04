
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
#include "wavefunction.hpp"
#include "defines.h"
#include "gp_kernels.h"
#include "helper_cudagl.h"

namespace gpcore {
  Wavefunction Psi;
  int wWidth = std::max(512, DIM);
  int wHeight = std::max(512, DIM);
}

void evolve(int sizex, int sizey, cuDoubleComplex *arr) {
  int speed = 6000;
  int print = 5;

  printf("\n\n Starting GP... \n\n");

  render::startOpenGL();

  // // Create Wavefunction
  gpcore::Psi.Initialize(arr);


  for( int a = 0; a < 200; a++ ) {
      double mult = exp(-(double)a / speed);
      gpcore::Psi.Step(mult); 
      if (a % print == 0){
        glutMainLoopEvent();
      }
   }

  gpcore::Psi.ExportToVariable(arr);
  gpcore::Psi.Cleanup();
  render::cleanup();

  printf("\n Done \n\n");

  // return gpcore::hostPsi;
}