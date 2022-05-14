# Dockerfile for mmsegmentation

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

```
./launch_container_x11.sh <mmseg_version>
```

If you use mmsegmentation `0.24.1`, please type the following command.

```
./launch_container_x11.sh 0.24.1
```
