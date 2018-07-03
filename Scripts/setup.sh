#!/bin/bash
cd ..
wget -O model_data/weights/darknet.weights https://pjreddie.com/media/files/yolov3.weights
python utils/convert.py --config-file yolov3.cfg \
			--weights-file yolov3.weights
