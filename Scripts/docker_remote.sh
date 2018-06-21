XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | sudo xauth -f $XAUTH nmerge -
sudo chmod 777 $XAUTH
X11PORT=`echo $DISPLAY | sed 's/^[^:]*:\([^\.]\+\).*/\1/'`
TCPPORT=`expr 6000 + $X11PORT`
sudo ufw allow from 172.17.0.0/16 to any port $TCPPORT proto tcp
DISPLAY=`echo $DISPLAY | sed 's/^[^:]*\(.*\)/172.17.0.1\1/'`
sudo nvidia-docker run -ti --rm -e DISPLAY=$DISPLAY -v $XAUTH:$XAUTH \
   -e XAUTHORITY=$XAUTH -v /home/khoi/yolov3:/yolov3 -v /media/vebits/D/data/BDD/bdd100k:/yolov3/data/BDD nguyenkh001/vebits-yolov3:latest
