# 标准与模型库

## 共同设定

- 我们默认使用 4 卡分布式训练
- 所有 PyTorch 风格的 ImageNet 预训练网络由我们自己训练，和 [论文](https://arxiv.org/pdf/1812.01187.pdf) 保持一致。
  我们的 ResNet 网络是基于 ResNetV1c 的变种，在这里输入层的 7x7 卷积被 3个 3x3 取代
- 为了在不同的硬件上保持一致，我们以 `torch.cuda.max_memory_allocated()` 的最大值作为 GPU 占用率，同时设置 `torch.backends.cudnn.benchmark=False`。
  注意，这通常比 `nvidia-smi` 显示的要少
- 我们以网络 forward 和后处理的时间加和作为推理时间，除去数据加载时间。我们使用脚本 `tools/benchmark.py` 来获取推理时间，它在 `torch.backends.cudnn.benchmark=False` 的设定下，计算 200 张图片的平均推理时间
- 在框架中，有两种推理模式
  - `slide` 模式（滑动模式）：测试的配置文件字段 `test_cfg` 会是 `dict(mode='slide', crop_size=(769, 769), stride=(513, 513))`.
    在这个模式下，从原图中裁剪多个小图分别输入网络中进行推理。小图的大小和小图之间的距离由 `crop_size` 和 `stride` 决定，重合区域会进行平均
  - `whole` 模式 （全图模式）：测试的配置文件字段 `test_cfg` 会是 `dict(mode='whole')`. 在这个模式下，全图会被直接输入到网络中进行推理。
    对于 769x769 下训练的模型，我们默认使用 `slide` 进行推理，其余模型用 `whole` 进行推理
- 对于输入大小为 8x+1 （比如769），我们使用 `align_corners=True`。其余情况，对于输入大小为 8x (比如 512，1024)，我们使用 `align_corners=False`

## 基线

### FCN

请参考 [FCN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fcn) 获得详细信息。

### PSPNet

请参考 [PSPNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/pspnet) 获得详细信息。

### DeepLabV3

请参考 [DeepLabV3](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3) 获得详细信息。

### PSANet

请参考 [PSANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/psanet) 获得详细信息。

### DeepLabV3+

请参考 [DeepLabV3+](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3plus) 获得详细信息。

### UPerNet

请参考 [UPerNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/upernet) 获得详细信息。

### NonLocal Net

请参考 [NonLocal Net](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/nlnet) 获得详细信息。

### EncNet

请参考 [EncNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/encnet) 获得详细信息。

### CCNet

请参考 [CCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ccnet) 获得详细信息。

### DANet

请参考 [DANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/danet) 获得详细信息。

### APCNet

请参考 [APCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/apcnet) 获得详细信息。

### HRNet

请参考 [HRNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/hrnet) 获得详细信息。

### GCNet

请参考 [GCNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/gcnet) 获得详细信息。

### DMNet

请参考 [DMNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dmnet) 获得详细信息。

### ANN

请参考 [ANN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ann) 获得详细信息。

### OCRNet

请参考 [OCRNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/ocrnet) 获得详细信息。

### Fast-SCNN

请参考 [Fast-SCNN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/fastscnn) 获得详细信息。

### ResNeSt

请参考 [ResNeSt](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/resnest) 获得详细信息。

### Semantic FPN

请参考 [Semantic FPN](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/semfpn) 获得详细信息。

### PointRend

请参考 [PointRend](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/point_rend) 获得详细信息。

### MobileNetV2

请参考 [MobileNetV2](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2) 获得详细信息。

### MobileNetV3

请参考 [MobileNetV3](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v3) 获得详细信息。

### EMANet

请参考 [EMANet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/emanet) 获得详细信息。

### DNLNet

请参考 [DNLNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/dnlnet) 获得详细信息。

### CGNet

请参考 [CGNet](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/cgnet) 获得详细信息。

### Mixed Precision (FP16) Training

请参考 [Mixed Precision (FP16) Training 在 BiSeNetV2 训练的样例](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/bisenetv2/bisenetv2_fcn_fp16_4x4_1024x1024_160k_cityscapes.py) 获得详细信息。

## 速度标定（待更新）

### 硬件

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### 软件环境

- Python 3.7
- PyTorch 1.5
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08

### 训练速度

为了公平比较，我们全部使用 ResNet-101V1c 进行标定。输入大小为 1024x512，批量样本数为 2。

训练速度如下表，指标为每次迭代的时间，以秒为单位，越低越快。

| Implementation                                                              | PSPNet (s/iter) | DeepLabV3+ (s/iter) |
| --------------------------------------------------------------------------- | --------------- | ------------------- |
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)              | **0.83**        | **0.85**            |
| [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)                  | 0.84            | 0.85                |
| [CASILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch) | 1.15            | N/A                 |
| [vedaseg](https://github.com/Media-Smart/vedaseg)                           | 0.95            | 1.25                |

注意：DeepLabV3+ 的输出步长为 8。
