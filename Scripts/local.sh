#!/bin/sh
# Source code
YOLOv3=$(cd $(dirname "$0") && cd "../" && pwd)
SRC=/home/yolov3
# Data
DATA=/media/vebits/D/data/yolov3-keras
DATA_DOCKER=$SRC/data
COCO=/media/vebits/D/data/COCO
COCO_DOCKER=$SRC/data/COCO
# Model data
MODEL=/media/vebits/D/model_data/yolov3-keras
MODEL_DOCKER=$SRC/model_data
# Run docker container
xhost +local:docker
sudo nvidia-docker run --rm -ti \
			-p 0.0.0.0:8888:8080 \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-e DISPLAY=$DISPLAY \
			-v $YOLOv3:$SRC \
			-v $DATA:$DATA_DOCKER \
			-v $MODEL:$MODEL_DOCKER \
			-v $COCO:$COCO_DOCKER \
			-w $SRC \
			nguyenkh001/vebits-yolov3:latest
xhost -local:docker
