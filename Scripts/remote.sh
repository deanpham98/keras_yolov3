XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | sudo xauth -f $XAUTH nmerge -
sudo chmod 777 $XAUTH
X11PORT=`echo $DISPLAY | sed 's/^[^:]*:\([^\.]\+\).*/\1/'`
TCPPORT=`expr 6000 + $X11PORT`
sudo ufw allow from 172.17.0.0/16 to any port $TCPPORT proto tcp
DISPLAY=`echo $DISPLAY | sed 's/^[^:]*\(.*\)/172.17.0.1\1/'`
# Source code
YOLOv3=$(cd $(dirname "$0") && cd "../" && pwd)
SRC=/home/yolov3
# Data
DATA=/media/vebits/D/data/yolov3-keras
DATA_DOCKER=$SRC/data
COCO=/media/vebits/D/data/COCO
COCO_DOCKER=$SRC/data/COCO
# Model data
MODEL=/media/vebits/D/model_data
MODEL_DOCKER=$SRC/model_data
# Run docker container
sudo nvidia-docker run -ti --rm \
			-e DISPLAY=$DISPLAY \
			-v $XAUTH:$XAUTH \
			-e XAUTHORITY=$XAUTH \
			-v $YOLOv3:$SRC \
			-v $DATA:$DATA_DOCKER \
			-v $MODEL:$MODEL_DOCKER \
			-v $COCO:$COCO_DOCKER \
			-p 0.0.0.0:8888:8080 \
			-w $SRC \
			nguyenkh001/vebits-yolov3:latest
