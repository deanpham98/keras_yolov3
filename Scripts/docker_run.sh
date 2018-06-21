#!/bin/sh
xhost +local:docker
YOLOv3=$(cd $(dirname "$0") && cd "../" && pwd)
DATA='/media/vebits/D/data/BDD/bdd100k'
sudo nvidia-docker run --rm -ti -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v $YOLOv3:/yolov3 -v $DATA:/yolov3/data/BDD -p 0.0.0.0:8888:8080 nguyenkh001/vebits-yolov3:latest
xhost -local:docker
