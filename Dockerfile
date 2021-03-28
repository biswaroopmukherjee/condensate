# Use NVIDIA/CUDAGL as a parent image
FROM nvidia/cudagl:10.0-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    cmake \
    swig \
    libglew-dev \
    freeglut3-dev  \
    ffmpeg

# Install any needed packages specified in requirements.txt
RUN python3 -m pip --no-cache-dir install \
    jupyter \
    numpy \
    matplotlib \
    h5py \
    pandas \ 
    scipy \
    scikit-image \
    tqdm \ 
    Pillow \
    jupyter_contrib_nbextensions 

RUN jupyter contrib nbextension install --user \
    && jupyter nbextension enable toc2/main

# Make a symlink to numpy header files for SWIG
RUN ln -s /usr/local/lib/python3.6/dist-packages/numpy/core/include/numpy /usr/include/numpy

# Copy the current directory contents into the container at /gp
COPY . /gp

# Set the working directory to /gp
WORKDIR /gp

# Build condensate
RUN cd condensate/core \
    && rm -rf build \
    && mkdir build \
    && cd build \
    && cmake ..\
    && make

EXPOSE 8888


CMD ["bash", "-c", "jupyter notebook --notebook-dir=/gp/notebooks --ip 0.0.0.0 --no-browser --allow-root"]