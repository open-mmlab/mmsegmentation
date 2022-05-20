# Dockerfile for mmsegmentation

## Requirements

- NVIDIA graphics driver
- docker >=19.03
- nvidia-docker2

## Build docker image

```
./build_image.sh <mmseg_version>
```

If you use mmsegmentation `0.24.1`, please type the following command.

```
./build_image.sh 0.24.1
```

## Launch docker container

### without X11

```
./launch_container.sh <mmseg_version>
```

If you use mmsegmentation `0.24.1`, please type the following command.

```
./launch_container.sh 0.24.1
```

### with X11

Official [demo scripts](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1/demo) need X11. So, if you would like to visualize the results, please run the script `launch_container_x11.sh` to enable X11 in docker container.

```
./launch_container_x11.sh <mmseg_version>
```

If you use mmsegmentation `0.24.1`, please type the following command.

```
./launch_container_x11.sh 0.24.1
```
