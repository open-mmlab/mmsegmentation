# NPU (华为 昇腾)

## 使用方法

请参考 [MMCV 的安装文档](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) 来安装 NPU 版本的 MMCV。

以下展示单机四卡场景的运行指令:

```shell
bash tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py 4
```

以下展示单机单卡下的运行指令:

```shell
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py
```

## 模型验证结果

|        Model        | mIoU  | Config                                                                                                                                     | Download                                                                                                                                    |
| :-----------------: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
|   [deeplabv3](<>)   | 78.85 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py)         | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024_20230115_205626.json)     |
| [deeplabv3plus](<>) | 79.23 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py) | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024_20230116_043450.json) |
|     [hrnet](<>)     | 78.1  | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/hrnet/fcn_hr18_4xb2-40k_cityscapes-512x1024.py)                     | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/fcn_hr18_4xb2-40k_cityscapes-512x1024_20230116_215821.json)             |
|      [fcn](<>)      | 74.15 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py)                     | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/fcn_r50-d8_4xb2-40k_cityscapes-512x1024_20230111_083014.json)           |
|     [icnet](<>)     | 69.25 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/icnet/icnet_r50-d8_4xb2-80k_cityscapes-832x832.py)                  | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/icnet_r50-d8_4xb2-80k_cityscapes-832x832_20230119_002929.json)          |
|    [pspnet](<>)     | 77.21 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/pspnet/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024.py)              | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/pspnet_r50b-d8_4xb2-80k_cityscapes-512x1024_20230114_042721.json)       |
|     [unet](<>)      | 68.86 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py)              | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024_20230129_224750.json)     |
|    [upernet](<>)    | 77.81 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/upernet/upernet_r50_4xb2-40k_cityscapes-512x1024.py)                | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/upernet_r50_4xb2-40k_cityscapes-512x1024_20230129_014634.json)          |
|    [apcnet](<>)     | 78.02 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py)               | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024_20230209_212545.json)        |
|   [bisenetv1](<>)   | 76.04 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/bisenetv1/bisenetv1_r50-d32_4xb4-160k_cityscapes-1024x1024.py)      | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/bisenetv1_r50-d32_4xb4-160k_cityscapes-1024x1024_20230201_023946.json)  |
|   [bisenetv2](<>)   | 72.44 | [config](https://github.com/open-mmlab/mmsegmentation/tree/1.x/configs/bisenetv2/bisenetv2_fcn_4xb4-amp-160k_cityscapes-1024x1024.py)      | [log](https://download.openmmlab.com/mmsegmentation/v0.5/device/npu/bisenetv2_fcn_4xb4-amp-160k_cityscapes-1024x1024_20230205_215606.json)  |

**注意:**

- 如果没有特别标记，NPU 上的使用混合精度训练的结果与使用 FP32 的 GPU 上的结果相同。

**以上模型结果由华为昇腾团队提供**
