#ifndef __GP_KERNELS_H_
#define __GP_KERNELS_H_

#include <cuda.h>
#include <cufft.h>




// Texture pitch
extern size_t tPitch;
struct uchar4;

void setup_texture(int x, int y);
void bind_texture(void);
void unbind_texture(void);
void delete_texture(void);
void update_texture(cuDoubleComplex *data, size_t w, size_t h, size_t pitch);
void colormapKernelLauncher(uchar4 *d_out, cuDoubleComplex *devPsi, double scale, int w, int h);
void multKernelLauncher(cuDoubleComplex *devPsi, double mult, int w, int h);
void realspaceKernelLauncher(cuDoubleComplex *devPsi, cuDoubleComplex *devExpPotential, cuDoubleComplex *out,
                            double g, double dt, double useReal, double cooling, int w, int h);
void momentumspaceKernelLauncher(cuDoubleComplex *devPsi, cuDoubleComplex *devExpKinetic, cuDoubleComplex *out, int w, int h);
void gaugefieldKernelLauncher(double omega, 
                              double *devXkY, double *devYkX, cuDoubleComplex *devExpXkY, cuDoubleComplex *devExpYkX,
                              double dt, double useReal, double cooling, int w, int h);
void parSum(cuDoubleComplex *devPsi, double *density, double dx, int w, int h);

#endif

