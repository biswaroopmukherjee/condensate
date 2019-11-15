#ifndef DEFINES_H
#define DEFINES_H

#define DIM    512       // Square size of solver domain
#define DS    (DIM*DIM)  // Total domain size
#define CPADW (DIM/2+1)  // Padded width for real->complex in-place FFT
#define RPADW (2*(DIM/2+1))  // Padded width for real->complex in-place FFT
#define PDS   (DIM*CPADW) // Padded total domain size

#define DT     0.09f     // Delta T for interative solver

#define TILEX 32 // Tile width
#define TILEY 32 // Tile height

#endif
