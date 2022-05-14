#!/bin/sh

if [ $# != 1 ]; then
    echo "usage: $0 <mmseg_version>"
    exit 1
fi

mmseg_version=$1

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run --gpus all -it \
           --volume=$XSOCK:$XSOCK:rw \
           --volume=$XAUTH:$XAUTH:rw \
           --shm-size=4gb \
           --env="XAUTHORITY=${XAUTH}" \
           --env="DISPLAY=${DISPLAY}" \
           --env=TERM=xterm-256color \
           --env=QT_X11_NO_MITSHM=1 \
           --net=host \
           --name=mmsegmentation-${mmseg_version}-x11 \
           mmsegmentation:${mmseg_version} \
           bash
