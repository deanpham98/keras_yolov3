#!/bin/sh
xhost +local:docker
YOLOv3=$(cd $(dirname "$0") && cd "../" && pwd)
sudo nvidia-docker run --rm -ti -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v $YOLOv3:/yolov3 -p 0.0.0.0:8888:8080 nguyenkh001/vebits-yolov3:latest
xhost -local:docker
