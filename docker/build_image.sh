#!/bin/sh

if [ $# != 1 ]; then
    echo "usage: $0 <mmseg_version>"
    exit 1
fi

mmseg_version=$1

docker build -t mmsegmentation:${mmseg_version} \
             --build-arg GID=$(id -g) \
             --build-arg UID=$(id -u) \
             --build-arg MMSEG_VERISON=v${mmseg_version} \
             .
