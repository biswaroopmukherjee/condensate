#!/bin/bash
# This is the startup script for condensate.

DOCKER_BUILDKIT=1 docker build --tag=gp .
xhost +local:docker

docker container rm condensate 
docker system prune -f

docker run -it \
    --name condensate \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --runtime=nvidia \
    --volume $(pwd)/notebooks:/gp/notebooks \
    -p 8888:8888\
    gp 
