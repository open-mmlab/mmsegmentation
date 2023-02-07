# NPU (华为 昇腾)

## 使用方法

请参考 [MMCV 的安装文档](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-mmcv-full-on-ascend-npu-machine) 来安装 NPU 版本的 MMCV。

以下展示单机八卡场景的运行指令:

```shell
bash tools/dist_train.sh configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py 4
```

以下展示单机单卡下的运行指令:

```shell
python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py
```

## 模型验证结果

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

**注意:**

- 如果没有特别标记，NPU 上的使用混合精度训练的结果与使用 FP32 的 GPU 上的结果结果相同。

**以上模型结果由华为昇腾团队提供**
