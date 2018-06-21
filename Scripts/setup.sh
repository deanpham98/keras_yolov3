#!/bin/bash
cd ..
mkdir data/annotations
mkdir models
mkdir models/weights
mkdir models/logs
mkdir models/checkpoints
wget https://pjreddie.com/media/files/yolov3.weights
python setup/convert.py config/yolov3.cfg yolov3.weights models/yolov3.h5
rm yolov3.weights


