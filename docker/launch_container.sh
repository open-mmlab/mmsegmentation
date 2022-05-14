#!/bin/sh

if [ $# != 1 ]; then
    echo "usage: $0 <mmseg_version>"
    exit 1
fi

mmseg_version=$1

docker run --gpus all -it \
           --shm-size=4gb \
           --env=TERM=xterm-256color \
           --net=host \
           --name=mmsegmentation-${mmseg_version} \
           mmsegmentation:${mmseg_version} \
           bash
