#!/bin/bash
# This is the startup script for condensate.

if [ $# = 1 ]
then

    if [ $1 == "docker" ]
    then
        DOCKER_BUILDKIT=1 docker build --tag=gp .
        xhost +local:docker

        docker system prune -f

        docker run -it --rm\
            --name condensate \
            --env="DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --gpus all \
            --volume $(pwd)/notebooks:/gp/notebooks \
            -p 8888:8888\
            gp 

    elif [ $1 == "leap" ]
    then
        cd condensate/core
        mkdir -p build
        ./build.sh
        cd ../..
        gnome-terminal -- /bin/sh -c 'sudo leapd; exec bash'
        jupyter notebook
    fi

else
    cd condensate/core
    mkdir -p build
    ./build.sh
    cd ../..
    jupyter notebook
fi
