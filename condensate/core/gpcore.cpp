
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
#include "chamber.hpp"
#include "defines.h"
#include "gp_kernels.h"
#include "helper_cudagl.h"


namespace gpcore {
  Chamber chamber;
  Wavefunction Psi;
}


// Set the spatial grid size of the system
void setup(int size, double deltat, double time, double omega_r, bool useImag, double cool) {
  gpcore::chamber.setup(size, deltat, time, omega_r, useImag, cool);
}


// Set the parameters of the harmonic potential, and use that as the current potential
void setHarmonicPotential(double omega, double epsilon) {
  gpcore::chamber.setHarmonicPotential(omega, epsilon);
}


// Extract the potential. Todo: import potential, timedependent: make an object, like wavefunction
void getPotential(int sizeX, int sizeY, double *V){
  unsigned int i;
  for( i=0; i<gpcore::chamber.DS; i++ ) {
    V[i] = gpcore::chamber.Potential[i] / HBAR;
  }
}


// evolve the wavefunction
void Evolve(int sizex, int sizey, cuDoubleComplex *arr, int skip) {

  printf("\n\n Starting GP... \n\n");
  render::startOpenGL();
  gpcore::Psi.Initialize(arr);

  for( int a = 0; a < gpcore::chamber.timesteps; a++ ) {

      gpcore::Psi.RealSpaceHalfStep(); 
      gpcore::Psi.MomentumSpaceStep();
      gpcore::Psi.RealSpaceHalfStep();
      gpcore::Psi.Renormalize();
      // gpcore::Psi.RotatingFrame(gpcore::chamber.omega[a]);
      if (a % skip ==0) glutMainLoopEvent();
   }

  gpcore::Psi.ExportToVariable(arr);
  gpcore::Psi.Cleanup();
  gpcore::chamber.Cleanup();
  render::cleanup();

  printf("\n Done \n\n");
}