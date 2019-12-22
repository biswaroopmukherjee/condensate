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
void kernelLauncher(uchar4 *d_out, cuDoubleComplex *devPsi, int w, int h);
void multKernel(cuDoubleComplex *devPsi, double mult, int w, int h);
#endif
