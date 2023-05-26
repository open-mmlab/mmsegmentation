# NPU (华为昇腾)

## 使用方法

首先，请参考[MMCV](https://mmcv.readthedocs.io/zh_CN/latest/get_started/build.html#npu-mmcv-full) 安装带有 NPU 支持的 MMCV与 [MMEngine](https://mmengine.readthedocs.io/en/latest/get_started/installation.html#build-from-source) 。
使用如下命令，可以利用 4 个 NPU 训练模型（以 deeplabv3为例）：

```shell
bash tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py 4
```

或者，使用如下命令，在一个 NPU 上训练模型（以 deeplabv3为例）：

```shell
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py
```

## 经过验证的模型

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

**注意:**

- 如果没有特别标记，NPU 上的结果与使用 FP32 的 GPU 上的结果结果相同。

**以上所有模型权重及训练日志均由华为昇腾团队提供**
