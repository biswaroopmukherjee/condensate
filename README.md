# condensate
Interactive GPU-accelerated numerical solutions of the GP equation. Written in C++/CUDA, wrapped in python using SWIG.

![condensate](https://s6.gifyu.com/images/condensate.gif)

## Usage

This project only runs on Linux machines with an NVIDIA GPU. Docker makes it easy to get started. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (make sure you have the official NVIDIA drivers). Then, run in your terminal

```bash
chmod +x start-condensate.sh
./start-condensate.sh docker
```

Note: the Leap Motion controller is currently incompatible with Docker. If you want to use the leap motion, replace `docker` with `leap` in the above commands. Leave out the command if you want to build locally without docker or leap. For these options, you need to install CUDA 10.0 or greater, the [V2 Leap motion SDK](https://developer.leapmotion.com/setup/desktop) for Linux, and a few other requirements that can be gleaned from the Dockerfile.

An overview of the python notebooks:

- The [Vortex Lattices](notebooks/Vortex%20Lattices.ipynb)  notebook is a good starting point with rotating gases.
- The [Leap motion control](notebooks/Leap%20motion%20control.ipynb) notebook lets you interact with a condensate.
- Finally, the [gpcore experiments](notebooks/gpcore%20experiments.ipynb) notebook goes into a few experiments we've been trying out, with the bare python bindings.

In order to run a simple script with an interactive condensate (instead of the notebooks), run `python3 scripts/mousetest.py` from either your docker container or if you have CUDA, directly in your terminal.

## Project overview

We use a time splitting Fourier pseudospectral (TSSP) method to solve the time-dependent GP equation in 2D. See [here](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/4/11813/files/2019/05/becreview.pdf) for more mathematical details. Note that there have been [many](http://gpelab.math.cnrs.fr/) [other](https://gpue-group.github.io/) implementations of a GP solver. 

The spirit of this project is to combine rapid experimentation with raw GPU compute performance. To this end, the heavy lifting is done by the CUDA kernels and C++ objects defined in `condensate/core`. In order to see the density in realtime, the wavefunction is rendered with an OpenGL pixelbuffer object. Since the rendering is done at the buffer refresh rate (tens of Hz) on the GPU, the performance overhead from the rendering is negligible. Rendering on the GPU is much faster than copying the wavefunction from the GPU to the host (CPU), and then to the disk for visualization. In order to access the C++ methods in a flexible manner, we use [SWIG](http://www.swig.org/) to generate python bindings. These are then called from jupyter notebook, and allows us to use a very simple API to run experiments.

