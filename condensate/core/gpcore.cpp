
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
#include "leap_controls.hpp"

namespace gpcore {
  Chamber chamber;
  Wavefunction Psi;
}

// Set the spatial grid size of the system
void Setup(int size, double fov, int size_g, double *g, double deltat, bool useImag, double cool, bool every_time_reset_potential) {
  gpcore::chamber.setup(size, fov, size_g, g, deltat, useImag, cool, every_time_reset_potential);
}

// Set the parameters of the harmonic potential, and use that as the current potential
void SetHarmonicPotential(int size_o, double *omega_t, int size_e, double *epsilon_t) {
    int size = size_o;
    gpcore::chamber.omega = (double *) malloc(sizeof(double) * size);
    gpcore::chamber.epsilon = (double *) malloc(sizeof(double) * size);
    for( unsigned int i=0; i<size; i++ ) {
    gpcore::chamber.omega[i] = omega_t[i];
    gpcore::chamber.epsilon[i] = epsilon_t[i];
    }
    gpcore::chamber.setHarmonicPotential(omega_t[0], epsilon_t[0]);
}

// Set parameters of a circular edge potential
void SetEdgePotential(double strength, double radius, double sharpness){gpcore::chamber.setEdgePotential(strength, radius, sharpness);}

// Extract the potential. Todo:  timedependent: make an object, like wavefunction
void GetPotential(int sizeX, int sizeY, double *V){
  for( unsigned int i=0; i<gpcore::chamber.DS; i++ ) {
    V[i] = gpcore::chamber.Potential[i] / HBAR;
  }
  gpcore::chamber.setupPotential();
}

// import potential
void SetPotential(int sizeX, int sizeY, double *V){
  for( unsigned int i=0; i<gpcore::chamber.DS; i++ ) {
    gpcore::chamber.Potential[i] = V[i] * HBAR;
  }
}
// Setup rotating frame with varying rotation freq omegaR. 
void RotatingFrame(int size, double *omega_r){
  gpcore::chamber.useRotatingFrame = true;
  gpcore::chamber.omegaR = (double *) malloc(sizeof(double) * size);
  for( unsigned int i=0; i<size; i++ ) {
    gpcore::chamber.omegaR[i] = omega_r[i];
  }
};

// Set up absorbing boundary conditions at a radius
void AbsorbingBoundaryConditions(double strength, double radius) {
  gpcore::chamber.AbsorbingBoundaryConditions(strength, radius);
}

// Set up a spoon
void SetupSpoon(double strength, double radius) {gpcore::chamber.SetupSpoon(strength, radius);}

// Set up a leap motion tracker
void SetupLeapMotion(double centerx, double centery, double zoomx, double zoomy, bool controlstrength) {
  gpcore::chamber.useLeapMotion = true;
  gpcore::chamber.LeapProps = {centerx, centery, zoomx, zoomy};
  gpcore::chamber.useLeapZ = controlstrength;
}


// evolve the wavefunction
void Evolve(int sizex, int sizey, cuDoubleComplex *arr, 
            unsigned long steps, int skip, bool show, double vmax, char *filename) {
  std::cout << "\nStarting GP..." << std::endl;
  gpcore::Psi.Initialize(arr);

  bool movie = std::string(filename).length()>0;
  if (movie) gpcore::Psi.InitializeMovie(filename);

  if (show) render::startOpenGL();
  if (show || movie) gpcore::chamber.cmapscale = vmax;
  Controller controller;

  unsigned long a=0;
  while (!gpcore::chamber.stopSim) 
  {
    gpcore::Psi.RealSpaceHalfStep(a); 
    gpcore::Psi.MomentumSpaceStep();
    gpcore::Psi.RealSpaceHalfStep(a);
    if (gpcore::chamber.useRotatingFrame) gpcore::Psi.RotatingFrame(a, steps);
    if (gpcore::chamber.every_time_reset_potential && (a != 0)) gpcore::chamber.TimeVary(a);
    if (gpcore::chamber.useImaginaryTime || (gpcore::chamber.cooling!=0)) gpcore::Psi.Renormalize();
    if (gpcore::chamber.spoon1.strength != 0) gpcore::chamber.Spoon();
    if (gpcore::chamber.useLeapMotion) LeapControl(controller);
    if (a%skip == 0){
      if (movie) gpcore::Psi.MovieFrame();
      if (show) glutMainLoopEvent();
    } ;
    a++;
    if (a==steps) gpcore::chamber.stopSim = true; 

   }

  
  if (show) render::cleanup();
  if (movie) gpcore::Psi.MovieCleanup();
  // gpcore::Psi.CalculateEnergy(energy);
  gpcore::Psi.ExportToVariable(arr);
  gpcore::Psi.Cleanup();
  gpcore::chamber.Cleanup();

  std::cout << "Done\n" << std::endl;
}
