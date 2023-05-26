# NPU (HUAWEI Ascend)

## Usage

Please refer to the [building documentation of MMCV](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) to install MMCV and [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html#build-from-source) on NPU devices.

Here we use 4 NPUs on your computer to train the model with the following command:

```shell
bash tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4
```

Also, you can use only one NPU to train the model with the following command:

```shell
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py
```

## Models Results

|        Model        | mIoU  | Config                                                                                                                                   | Download                                                                                                                   |
| :-----------------: | :---: | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
|   [deeplabv3](<>)   | 78.92 | [config](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py)         | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/deeplabv3_r50-d8_512x1024_40k_cityscapes.log.json)     |
| [deeplabv3plus](<>) | 79.68 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py) | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.log.json) |
|     [hrnet](<>)     | 77.09 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py)                     | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/fcn_hr18_512x1024_40k_cityscapes.log.json)             |
|      [fcn](<>)      | 72.69 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py)                     | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/fcn_r50-d8_512x1024_40k_cityscapes.log.json)           |
|    [pspnet](<>)     | 78.07 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py)               | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/pspnet_r50-d8_512x1024_80k_cityscapes.log.json)        |
|     [unet](<>)      | 69.00 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py)          | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.log.json) |
|    [apcnet](<>)     | 78.07 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py)               | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/apcnet_r50-d8_512x1024_40k_cityscapes.log.json)        |
|    [upernet](<>)    | 78.15 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/upernet/upernet_r50_512x1024_40k_cityscapes.py)                | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/20230525_064810.log.json)                              |

**Notes:**

- If not specially marked, the results on NPU with amp are the basically same as those on the GPU with FP32.

**All above models are provided by Huawei Ascend group.**
