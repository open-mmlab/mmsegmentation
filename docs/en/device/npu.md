# NPU (HUAWEI Ascend)

## Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV on NPU devices

Here we use 4 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py 4
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py
```

## Models Results

|        Model        | mIoU  | Config                                                                                                                                | Download  |
| :-----------------: | :---: | :------------------------------------------------------------------------------------------------------------------------------------ | :-------- |
|   [deeplabv3](<>)   | 78.85 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py) | [log](<>) |
| [deeplabv3plus](<>) | 79.23 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024)     | [log](<>) |
|     [hrnet](<>)     | 78.1  | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/hrnet/fcn_hr18_4xb2-40k_cityscapes-512x1024.py)        | [log](<>) |
|      [fcn](<>)      | 74.15 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py)        | [log](<>) |
|     [icnet](<>)     | 69.25 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/icnet/icnet_r50-d8_4xb2-80k_cityscapes-832x832.py)     | [log](<>) |
|    [pspnet](<>)     | 77.21 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/pspnet/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py) | [log](<>) |
|     [unet](<>)      | 68.86 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py) | [log](<>) |
|    [upernet](<>)    | 77.81 | [config](https://github.com/wangjiangben-hw/mmsegmentation/blob/master/configs/upernet/upernet_r50_4xb2-40k_cityscapes-512x1024.py)   | [log](<>) |

**Notes:**

- If not specially marked, the results on NPU with amp are the basically same as those on the GPU with FP32.

**All above models are provided by Huawei Ascend group.**
